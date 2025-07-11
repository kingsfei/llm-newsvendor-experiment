import os
import time
import re
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.stats import truncnorm
from scipy import optimize
from openai import OpenAI

# Use environment variable to get API key (more secure)
client = OpenAI(
    api_key="your api"
)


# ========== Configuration ==========
GPT_MODEL = "gpt-4o"
TEMPERATURE = 0.0
NUM_TRIALS = 2
NUM_ROUNDS = 15  # 15 rounds per condition, 30 rounds total
PAUSE = 2  # Pause time after each request to avoid rate limits
MAX_RETRIES = 3  # Maximum retry attempts if extraction fails


# ========== Helper Functions ==========

def ask_gpt(prompt):
    """
    Call OpenAI GPT API and return the generated text.
    """
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system",
             "content": "You are participating in a decision-making experiment on inventory management."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


def extract_order_quantity(response_text):
    """
    Extract order quantity from GPT response, trying various possible formats
    """
    # Try multiple possible extraction patterns
    patterns = [
        r"(?i)final\s*order\s*quantity:\s*(\d+)",
        r"(?i)I\s*(would|will)?\s*order\s*(\d+)",
        r"(?i)order\s*quantity[:\s]*(\d+)",
        r"(?i)order\s*(\d+)",
        r"(?i)I\s*(would|will)?\s*purchase\s*(\d+)",
        r"(?i)purchase\s*(\d+)",
        r"(?i)decide\s*to\s*order\s*(\d+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            # Determine extraction group based on number of groups
            if len(match.groups()) == 1:
                quantity = int(match.group(1))
                return quantity
            elif len(match.groups()) == 2:
                quantity = int(match.group(2))
                return quantity

    # No match found, search for numbers directly
    numbers = re.findall(r'(\d+)', response_text)
    if numbers:
        for num in numbers:
            quantity = int(num)
            if quantity > 0:  # Avoid extracting irrelevant numbers like years
                return quantity

    return None


def calculate_profit(order_quantity, demand, price, cost, salvage):
    """
    Calculate profit for this round
    """
    sales = min(order_quantity, demand)
    revenue = price * sales
    salvage_revenue = salvage * max(0, order_quantity - demand)
    total_cost = cost * order_quantity
    return revenue + salvage_revenue - total_cost


def ask_gpt_with_retry(prompt, correction_prompt_template, max_retries=MAX_RETRIES):
    """
    First ask GPT with `prompt`, if unable to parse order quantity, use `correction_prompt_template`
    to remind GPT to correct answer multiple times until successful extraction or max retries reached.
    """
    gpt_response = ""
    for attempt in range(max_retries):
        if attempt == 0:
            # First attempt without correction prompt
            gpt_response = ask_gpt(prompt)
        else:
            # Subsequent attempts with correction prompt
            correction_prompt = correction_prompt_template.format(previous_answer=gpt_response)
            gpt_response = ask_gpt(correction_prompt)

        order_qty = extract_order_quantity(gpt_response)
        if order_qty is not None:
            return gpt_response, order_qty

    # If still unable to parse after repeated retries, default to 150
    print(f"[Warning] After {max_retries} attempts, no valid order quantity extracted. Default = 150.")
    return gpt_response, 150


# ========== Calculate Optimal Order Quantity ==========

def calculate_optimal_order(distribution_type, a, b, price, cost, salvage):
    """
    Calculate optimal order quantity under different distributions

    Parameters:
    - distribution_type: Distribution type ('uniform', 'normal', 'lognormal')
    - a, b: Lower and upper bounds of demand range
    - price: Price
    - cost: Cost
    - salvage: Salvage value

    Returns:
    - Optimal order quantity
    """
    # Calculate critical fractile
    critical_fractile = (price - cost) / (price - salvage)
    mean = (a + b) / 2
    std = (b - a) / 6  # Standard deviation approximation for uniform distribution

    if distribution_type == 'uniform':
        # Optimal order quantity for uniform distribution
        optimal_order = a + (b - a) * critical_fractile

    elif distribution_type == 'normal':
        # Calculate optimal order quantity using truncated normal distribution
        a_param = (a - mean) / std  # Lower truncation point
        b_param = (b - mean) / std  # Upper truncation point

        # Calculate optimal order quantity under truncated normal distribution
        optimal_order = truncnorm.ppf(critical_fractile, a_param, b_param, loc=mean, scale=std)

    elif distribution_type == 'lognormal':
        # Parameter optimization for lognormal distribution to have appropriate distribution characteristics in [a,b] interval
        def objective(params):
            mu, sigma = params
            # Calculate integral of lognormal distribution in [a,b] range
            prob_in_range = stats.lognorm.cdf(b, s=sigma, scale=np.exp(mu)) - stats.lognorm.cdf(a, s=sigma,
                                                                                                scale=np.exp(mu))
            # Calculate difference between mean and target
            dist_mean = np.exp(mu + 0.5 * sigma ** 2)  # Theoretical mean of lognormal distribution
            mean_diff = abs(dist_mean - mean)
            # Return a composite score (want distribution concentrated in range and mean close to expected)
            return (1 - prob_in_range) * 10 + mean_diff / mean

        # Optimize parameters, initial guess to make distribution mean close to (a+b)/2
        initial_mu = np.log(mean) - 0.5 * 0.5 ** 2  # Assume sigma=0.5, find mu to make mean equal to mean
        result = optimize.minimize(
            objective,
            [initial_mu, 0.5],
            bounds=[(None, None), (0.01, 2.0)],
            method='L-BFGS-B'
        )
        mu_opt, sigma_opt = result.x

        # Calculate distribution-adjusted quantile in [a,b] interval
        p_a = stats.lognorm.cdf(a, s=sigma_opt, scale=np.exp(mu_opt))
        p_b = stats.lognorm.cdf(b, s=sigma_opt, scale=np.exp(mu_opt))

        # Adjust critical fractile to [p_a, p_b] range
        adjusted_cf = p_a + (p_b - p_a) * critical_fractile

        # Calculate optimal order quantity
        optimal_order = stats.lognorm.ppf(adjusted_cf, s=sigma_opt, scale=np.exp(mu_opt))

        # Ensure within range
        optimal_order = np.clip(optimal_order, a, b)

        # Output some debugging information
        print(f"Lognormal distribution parameters: mu={mu_opt:.4f}, sigma={sigma_opt:.4f}")
        print(f"Probability of distribution in [{a},{b}] range: {(p_b - p_a) * 100:.2f}%")
        print(f"Original critical fractile: {critical_fractile:.4f}, adjusted: {adjusted_cf:.4f}")

    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

    # Ensure optimal order quantity is within valid range and rounded
    optimal_order = np.clip(optimal_order, a, b)
    return round(optimal_order)


# ========== Generate Demand Sequences for Different Distributions ==========

def get_demand_sequence(condition, distribution_type):
    """
    Generate demand sequence based on condition and distribution type
    """
    np.random.seed(42)  # Fixed random seed for reproducibility

    # Determine demand range
    if "high_range" in condition:
        a, b = 901, 1200
    else:
        a, b = 1, 300

    mean = (a + b) / 2  # Mean
    std = (b - a) / 6  # Standard deviation

    # Fixed demand sequence for Experiment 1
    if distribution_type == "uniform" and "exp1" in condition:
        if "high_profit" in condition:
            return [290, 260, 10, 270, 110, 155, 235, 10, 175, 150, 80, 175, 200, 70, 75]
        elif "low_profit" in condition:
            return [210, 160, 170, 80, 85, 240, 5, 220, 230, 190, 15, 130, 255, 230, 110]

    # Generate sequence based on distribution type
    if distribution_type == "uniform":
        return np.random.randint(a, b + 1, size=NUM_ROUNDS)

    elif distribution_type == "normal":
        # Generate demand using truncated normal distribution
        a_param = (a - mean) / std
        b_param = (b - mean) / std
        demands = truncnorm.rvs(a_param, b_param, loc=mean, scale=std, size=NUM_ROUNDS)
        return np.round(demands).astype(int)

    elif distribution_type == "lognormal":
        # Use same parameter optimization logic as in optimal order quantity calculation
        def objective(params):
            mu, sigma = params
            prob_in_range = stats.lognorm.cdf(b, s=sigma, scale=np.exp(mu)) - stats.lognorm.cdf(a, s=sigma,
                                                                                                scale=np.exp(mu))
            dist_mean = np.exp(mu + 0.5 * sigma ** 2)
            mean_diff = abs(dist_mean - mean)
            return (1 - prob_in_range) * 10 + mean_diff / mean

        initial_mu = np.log(mean) - 0.5 * 0.5 ** 2
        result = optimize.minimize(
            objective,
            [initial_mu, 0.5],
            bounds=[(None, None), (0.01, 2.0)],
            method='L-BFGS-B'
        )
        mu_opt, sigma_opt = result.x

        # Generate lognormal random numbers
        raw_demands = stats.lognorm.rvs(s=sigma_opt, scale=np.exp(mu_opt), size=NUM_ROUNDS)

        # Truncate if values exceed range
        demands = np.clip(raw_demands, a, b)
        return np.round(demands).astype(int)

    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")


# ========== Visualize Demand Distribution ==========

def visualize_demand_distribution(distribution_type, a, b):
    """
    Visualize demand distribution for different distribution types
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot load matplotlib, skipping visualization step")
        return

    # Generate 1000 sample points for each distribution type
    np.random.seed(42)

    mean = (a + b) / 2
    std = (b - a) / 6

    if distribution_type == "uniform":
        samples = np.random.randint(a, b + 1, size=1000)
        title = f"Uniform Distribution U[{a},{b}]"

    elif distribution_type == "normal":
        # Use truncated normal distribution
        a_param = (a - mean) / std
        b_param = (b - mean) / std
        samples = truncnorm.rvs(a_param, b_param, loc=mean, scale=std, size=1000)
        samples = np.round(samples).astype(int)
        title = f"Normal Distribution N({mean:.1f},{std:.1f}) [truncated at {a},{b}]"

    elif distribution_type == "lognormal":
        # Use same parameter optimization as get_demand_sequence
        def objective(params):
            mu, sigma = params
            prob_in_range = stats.lognorm.cdf(b, s=sigma, scale=np.exp(mu)) - stats.lognorm.cdf(a, s=sigma,
                                                                                                scale=np.exp(mu))
            dist_mean = np.exp(mu + 0.5 * sigma ** 2)
            mean_diff = abs(dist_mean - mean)
            return (1 - prob_in_range) * 10 + mean_diff / mean

        initial_mu = np.log(mean) - 0.5 * 0.5 ** 2
        result = optimize.minimize(
            objective,
            [initial_mu, 0.5],
            bounds=[(None, None), (0.01, 2.0)],
            method='L-BFGS-B'
        )
        mu_opt, sigma_opt = result.x

        # Generate lognormal random numbers
        raw_samples = stats.lognorm.rvs(s=sigma_opt, scale=np.exp(mu_opt), size=1000)
        samples = np.clip(raw_samples, a, b)
        samples = np.round(samples).astype(int)
        title = f"Lognormal Distribution (parameters: μ={mu_opt:.2f}, σ={sigma_opt:.2f})"

    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

    # Create save directory
    os.makedirs("visualizations", exist_ok=True)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=30, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Add distribution characteristics information
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)
    sample_median = np.median(samples)
    plt.annotate(f"Sample Mean: {sample_mean:.1f}\nSample Median: {sample_median:.1f}\nSample Std: {sample_std:.1f}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Save image
    filename = f"visualizations/demand_distribution_{distribution_type}_{a}_{b}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Distribution chart saved as: {filename}")


# ========== Create Experiment Prompt Templates ==========

def create_prompt_templates():
    """
    Create prompt templates for all experimental conditions
    """
    prompts = {}

    # Templates for Experiment 1 - three distribution types
    distributions = ["uniform", "normal", "lognormal"]
    profit_conditions = ["high_profit", "low_profit"]

    # Distribution type descriptions
    dist_desc = {
        "uniform": "demand is uniformly distributed between {a} and {b}",
        "normal": "demand follows a normal distribution with mean {mean} and standard deviation approximately {std}, mainly ranging between {a} and {b}",
        "lognormal": "demand follows a right-skewed distribution (similar to lognormal distribution) with mean approximately {mean}, having a longer right tail, ranging between {a} and {b}"
    }

    # Create templates for Experiment 1
    for dist in distributions:
        for profit in profit_conditions:
            # Determine cost
            cost = 3 if "high_profit" in profit else 9

            # For different demand ranges
            for exp_type in ["exp1", "exp2"]:
                for range_type in ["low_range", "high_range"] if exp_type == "exp2" else [""]:
                    # Determine demand range
                    a, b = (901, 1200) if range_type == "high_range" else (1, 300)
                    mean = (a + b) / 2
                    std = (b - a) / 6

                    # Generate condition name
                    if exp_type == "exp1":
                        condition = f"{exp_type}_{profit}_{dist}"
                    else:
                        condition = f"{exp_type}_{profit}_{range_type}_{dist}"

                    # Base prompt template
                    template = f"""
You are participating in an inventory decision experiment. You need to make inventory ordering decisions for selling "wodgets" products.

Experiment rules:
- Each wodget sells for 12 francs
- Each wodget costs {cost} francs
- Remaining unsold wodgets can be disposed of for 0 francs
- {dist_desc[dist].format(a=a, b=b, mean=mean, std=std)}
- Your goal is to maximize profit

{{history}}

Here is some information that might be helpful:
- If sales exceed your order quantity, you will lose some sales opportunities
- If order quantity exceeds sales, you will need to dispose of remaining inventory
- Please consider the costs of ordering too much and ordering too little
"""

                    # For Experiment 2, add content about optimal formula
                    if exp_type == "exp2":
                        template += """
Note: You have previously learned about the newsvendor problem. In this problem, the profit-maximizing order quantity can be determined by the following formula:
F(q*) = (p-c)/(p-s)
where F(q) is the cumulative distribution function, p is selling price, c is cost, s is salvage value.
"""
                        # Provide additional information for different distribution types
                        if dist == "uniform":
                            template += f"For uniform distribution U[{a},{b}], F(q) = (q-{a})/({b}-{a}).\n"
                        elif dist == "normal":
                            template += f"For normal distribution N({mean:.1f},{std:.1f}), use standard normal CDF for calculation.\n"
                        elif dist == "lognormal":
                            template += "For right-skewed distribution, calculate cumulative probability based on specific distribution characteristics.\n"

                    template += "\nPlease make an inventory ordering decision: How many wodgets will you order? Please explain your thinking process and decision in detail."

                    prompts[condition] = template

    return prompts


# ========== Correction Prompt ==========
CORRECTION_PROMPT = """
Your previous answer did not clearly indicate the quantity you want to order. Please answer again and clearly write in the last line of your answer:
"I finally decide to order X wodgets", where X is an integer.

Previous answer:
"{previous_answer}"
"""


# ========== Initialize Experiment Settings ==========
def initialize_experiment_settings():
    """
    Initialize parameter settings and optimal order quantities for all experimental conditions
    """
    settings = {}

    # Basic experimental parameter configuration
    distributions = ["uniform", "normal", "lognormal"]
    profit_conditions = [
        ("high_profit", 3),  # (condition name, cost)
        ("low_profit", 9)
    ]

    # All experimental conditions
    for exp_type in ["exp1", "exp2"]:
        for range_name, range_values in [("low_range", (1, 300)), ("high_range", (901, 1200))]:
            # Experiment 1 only has low demand range
            if exp_type == "exp1" and range_name == "high_range":
                continue

            a, b = range_values

            for profit_name, cost in profit_conditions:
                for dist in distributions:
                    # Generate condition name
                    if exp_type == "exp1":
                        condition = f"{exp_type}_{profit_name}_{dist}"
                    else:
                        condition = f"{exp_type}_{profit_name}_{range_name}_{dist}"

                    # Basic parameter settings
                    settings[condition] = {
                        "price": 12,
                        "cost": cost,
                        "salvage": 0,
                        "distribution": dist,
                        "demand_range": (a, b)
                    }

                    # Calculate optimal order quantity
                    optimal_order = calculate_optimal_order(
                        dist, a, b, 12, cost, 0
                    )

                    # Add to settings
                    settings[condition]["optimal_order"] = optimal_order

                    # Print information
                    print(f"Condition: {condition}, Optimal order quantity: {optimal_order}")

    return settings


# ========== Run Experiment ==========
def run_experiment(experiment_type, distribution_type, high_profit_first=True):
    """
    Run experiment for specified experiment type and distribution type

    Parameters:
    - experiment_type: 'exp1' or 'exp2'
    - distribution_type: 'uniform', 'normal', or 'lognormal'
    - high_profit_first: whether high profit condition comes first
    """
    data = []
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Determine demand range prefix (only valid for experiment 2)
    range_types = ["low_range", "high_range"] if experiment_type == "exp2" else [""]

    for range_type in range_types:
        # Determine condition order based on high_profit_first
        if high_profit_first:
            if experiment_type == "exp1":
                first_condition = f"{experiment_type}_high_profit_{distribution_type}"
                second_condition = f"{experiment_type}_low_profit_{distribution_type}"
            else:
                first_condition = f"{experiment_type}_high_profit_{range_type}_{distribution_type}"
                second_condition = f"{experiment_type}_low_profit_{range_type}_{distribution_type}"
        else:
            if experiment_type == "exp1":
                first_condition = f"{experiment_type}_low_profit_{distribution_type}"
                second_condition = f"{experiment_type}_high_profit_{distribution_type}"
            else:
                first_condition = f"{experiment_type}_low_profit_{range_type}_{distribution_type}"
                second_condition = f"{experiment_type}_high_profit_{range_type}_{distribution_type}"

        # Skip invalid conditions
        if first_condition not in EXPERIMENT_SETTINGS or second_condition not in EXPERIMENT_SETTINGS:
            continue

        for trial in range(NUM_TRIALS):
            print(
                f"\nStarting Trial {trial + 1}/{NUM_TRIALS} for {experiment_type} ({range_type}) with {distribution_type} distribution (High Profit First: {high_profit_first})")

            # Get demand sequences
            first_demand_sequence = get_demand_sequence(first_condition, distribution_type)
            second_demand_sequence = get_demand_sequence(second_condition, distribution_type)

            # Combine into one continuous 30-round demand sequence
            demand_sequence = np.concatenate([first_demand_sequence, second_demand_sequence])

            # Prepare condition settings
            first_settings = EXPERIMENT_SETTINGS[first_condition]
            second_settings = EXPERIMENT_SETTINGS[second_condition]

            cumulative_profit = 0
            history = ""
            last_order = None
            last_demand = None
            last_profit = None

            # Conduct continuous 30-round decisions
            for round_idx in range(NUM_ROUNDS * 2):
                # Determine which condition to use for current round
                if round_idx < NUM_ROUNDS:
                    current_condition = first_condition
                    current_settings = first_settings
                else:
                    current_condition = second_condition
                    current_settings = second_settings

                # Get demand for current round
                demand = demand_sequence[round_idx]

                print(f"  Round {round_idx + 1}/30, Condition: {current_condition}")

                if round_idx > 0:
                    # Build multi-round history information
                    history = f"""
In the previous round:
- Your order quantity: {last_order} wodgets
- Actual demand: {last_demand} wodgets
- This round's profit: {last_profit} francs
Current cumulative profit: {cumulative_profit} francs
"""

                # Construct prompt text
                prompt_text = PROMPT_TEMPLATES[current_condition].format(history=history)

                # Call GPT function with retry logic
                gpt_response, order_qty = ask_gpt_with_retry(prompt_text, CORRECTION_PROMPT)

                # Calculate profit
                profit = calculate_profit(
                    order_qty,
                    demand,
                    current_settings["price"],
                    current_settings["cost"],
                    current_settings["salvage"]
                )
                cumulative_profit += profit

                print(f"    Order: {order_qty}, Demand: {demand}, Profit: {profit}")

                # Save data
                data.append({
                    'Date': current_date,
                    'Model': GPT_MODEL,
                    'Temperature': TEMPERATURE,
                    'Experiment': experiment_type,
                    'RangeType': range_type if experiment_type == "exp2" else "low_range",
                    'HighProfitFirst': high_profit_first,
                    'Condition': current_condition,
                    'Distribution': distribution_type,
                    'Trial': trial + 1,
                    'Round': round_idx + 1,
                    'OrderQuantity': order_qty,
                    'Demand': demand,
                    'OptimalOrder': current_settings["optimal_order"],
                    'Price': current_settings["price"],
                    'Cost': current_settings["cost"],
                    'Salvage': current_settings["salvage"],
                    'Profit': profit,
                    'CumulativeProfit': cumulative_profit,
                    'GPTResponse': gpt_response
                })

                # Update previous round information
                last_order = order_qty
                last_demand = demand
                last_profit = profit

                time.sleep(PAUSE)

            # Save trial data immediately after each trial completion
            trial_df = pd.DataFrame(data)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            trial_filename = f"results/newsvendor-{experiment_type}-{distribution_type}-range{range_type}-highfirst{high_profit_first}-trial{trial + 1}-{GPT_MODEL}-{timestamp}.csv"

            # Ensure directory exists
            os.makedirs("results", exist_ok=True)
            trial_df.to_csv(trial_filename, index=False)
            print(f"    Trial {trial + 1} results saved to: {trial_filename}")

    return pd.DataFrame(data)


# ========== Main Function ==========
def main():
    print("Starting newsvendor problem experiment, based on Schweitzer and Cachon (2000) paper, extended with different distribution types!")
    print(f"Configuration: Model={GPT_MODEL}, Number of trials={NUM_TRIALS}, Rounds per condition={NUM_ROUNDS}\n")

    # Initialize experiment settings
    global EXPERIMENT_SETTINGS, PROMPT_TEMPLATES
    EXPERIMENT_SETTINGS = initialize_experiment_settings()
    PROMPT_TEMPLATES = create_prompt_templates()

    # Visualize demand distributions
    try:
        for dist in ["uniform", "normal", "lognormal"]:
            visualize_demand_distribution(dist, 1, 300)
            visualize_demand_distribution(dist, 901, 1200)
    except Exception as e:
        print(f"Unable to generate distribution visualization: {e}")

    # Experimental condition list
    experiment_conditions = [
        # Experiment 1 - Different distributions
        ("exp1", "uniform", True),
        ("exp1", "uniform", False),
        ("exp1", "normal", True),
        ("exp1", "normal", False),
        ("exp1", "lognormal", True),
        ("exp1", "lognormal", False),

        # Experiment 2 - Including two demand ranges, different distributions
        ("exp2", "uniform", True),
        ("exp2", "uniform", False),
        ("exp2", "normal", True),
        ("exp2", "normal", False),
        ("exp2", "lognormal", True),
        ("exp2", "lognormal", False)
    ]
    all_data = []
    # Create results save directory
    os.makedirs("results", exist_ok=True)

    # Run all experimental conditions sequentially
    for exp_type, dist_type, high_first in experiment_conditions:
        print(f"\n== Running {exp_type} (Distribution: {dist_type}, High Profit First: {high_first}) ==")
        df = run_experiment(exp_type, dist_type, high_profit_first=high_first)
        all_data.append(df)

        # Save results after each condition run to avoid data loss
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = f"results/newsvendor-{exp_type}-{dist_type}-highfirst{high_first}-{GPT_MODEL}-{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"  - Results saved to: {filename}")

    # Combine all results
    combined_df = pd.concat(all_data)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    all_filename = f"results/newsvendor-all-experiments-{GPT_MODEL}-{timestamp}.csv"
    combined_df.to_csv(all_filename, index=False)
    print(f"\nAll experiment results have been combined and saved to: {all_filename}")

if __name__ == "__main__":
    main()