# Large Language Newsvendor: Decision Biases and Cognitive Mechanisms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/api/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the experimental code for investigating cognitive biases and decision-making mechanisms in Large Language Models (LLMs) using the newsvendor problem. The study reveals the "Paradox of Intelligence" - where increased model complexity doesn't guarantee better decision-making.

## 📖 Overview

The newsvendor problem is a classical inventory management decision where an agent must determine optimal order quantities under demand uncertainty. This experiment investigates how three leading LLMs (GPT-4, GPT-4o, and LLaMA-8B) exhibit cognitive biases analogous to human decision-makers, including:

- **Systematic ordering bias** (underordering in high-profit, overordering in low-profit scenarios)
- **Bias persistence in risk-neutral environments**
- **Presentation-order effects** and path dependence
- **Demand-chasing behavior** with overreaction to recent signals
- **Constraints on learning from feedback**

### Key Findings

🔍 **The Paradox of Intelligence**: More sophisticated models don't always make better decisions
- GPT-4o (efficiency-optimized) → Near-optimal performance
- GPT-4 (complex reasoning) → Overthinking leads to suboptimal decisions  
- LLaMA-8B (resource-constrained) → Unstable heuristic behavior

📈 **Bias Amplification**: LLMs can amplify human biases by up to 70% beyond human benchmarks

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
openai >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
