#!/usr/bin/env python3
"""Quantitative reasoning, statistics, and mathematical analysis training data."""
import json

def generate_examples():
    examples = []

    # BAYESIAN REASONING
    bayesian = [
        "How do you update probability estimates with new evidence?",
        "What is Bayes' theorem and how does it apply to intelligence analysis?",
        "How should prior probabilities be established for geopolitical events?",
        "What is the difference between frequentist and Bayesian approaches?",
        "How do you handle uncertainty when evidence is conflicting?",
        "What is a likelihood ratio and how is it used in analysis?",
        "How do base rates affect probability estimates?",
        "What is the prosecutor's fallacy and how do you avoid it?",
        "How should analysts update beliefs when surprised by events?",
        "What role does conditional probability play in threat assessment?",
        "How do you quantify the strength of evidence?",
        "What is posterior probability and how is it calculated?",
        "How do you handle multiple competing hypotheses?",
        "What is the role of priors in intelligence estimation?",
        "How do you avoid anchoring bias in probability updates?",
    ]
    for q in bayesian:
        examples.append({"instruction": q, "input": "", "output": "[Bayesian analysis: prior establishment, evidence weighting, likelihood computation, posterior updating, and uncertainty quantification.]"})

    # TIME SERIES & FORECASTING
    timeseries = [
        "How do you identify trends vs. noise in time series data?",
        "What is autocorrelation and why does it matter for forecasting?",
        "How do you detect seasonality in economic indicators?",
        "What is the difference between ARIMA and exponential smoothing?",
        "How do you handle structural breaks in time series?",
        "What is stationarity and why is it important?",
        "How do you forecast with limited historical data?",
        "What is mean reversion and how do you model it?",
        "How do you combine multiple forecasting models?",
        "What is forecast error and how do you measure it?",
        "How do you detect regime changes in market data?",
        "What is the difference between interpolation and extrapolation?",
        "How do you handle missing data in time series?",
        "What is a leading indicator and how do you identify one?",
        "How do you model volatility clustering?",
    ]
    for q in timeseries:
        examples.append({"instruction": q, "input": "", "output": "[Time series analysis: trend detection, seasonality, stationarity testing, model selection, and forecast validation.]"})

    # MONTE CARLO & SIMULATION
    montecarlo = [
        "How do you use Monte Carlo simulation for scenario analysis?",
        "What distributions should be used for different risk factors?",
        "How many simulations are needed for reliable estimates?",
        "How do you model correlated risk factors in simulation?",
        "What is the difference between simulation and sensitivity analysis?",
        "How do you validate Monte Carlo model assumptions?",
        "What is importance sampling and when is it useful?",
        "How do you handle fat-tailed distributions in simulation?",
        "What is convergence and how do you test for it?",
        "How do you communicate simulation results to decision-makers?",
        "What is variance reduction and how is it achieved?",
        "How do you model rare events in simulation?",
        "What is the role of random seeds in reproducibility?",
        "How do you stress test simulation assumptions?",
        "What is bootstrapping and how does it relate to simulation?",
    ]
    for q in montecarlo:
        examples.append({"instruction": q, "input": "", "output": "[Monte Carlo analysis: distribution selection, correlation modeling, convergence testing, and result interpretation.]"})

    # RISK METRICS
    risk = [
        "What is Value at Risk (VaR) and how is it calculated?",
        "What are the limitations of VaR as a risk measure?",
        "What is Conditional VaR (CVaR) and when is it preferred?",
        "How do you measure tail risk in portfolios?",
        "What is drawdown and how is maximum drawdown calculated?",
        "How do you measure correlation breakdown during crises?",
        "What is the Sharpe ratio and what are its limitations?",
        "How do you measure systemic risk?",
        "What is the difference between volatility and risk?",
        "How do you measure concentration risk?",
        "What is liquidity risk and how is it quantified?",
        "How do you stress test risk models?",
        "What is model risk and how do you manage it?",
        "How do you aggregate risks across different types?",
        "What is the difference between expected and unexpected loss?",
    ]
    for q in risk:
        examples.append({"instruction": q, "input": "", "output": "[Risk metrics analysis: measurement methodology, limitations, stress testing, and practical application to decision-making.]"})

    # SIGNAL VS NOISE
    signal = [
        "How do you distinguish signal from noise in data?",
        "What is the signal-to-noise ratio and how is it measured?",
        "How do you filter noise without losing signal?",
        "What is overfitting and how do you avoid it?",
        "How do you detect spurious correlations?",
        "What is the role of sample size in signal detection?",
        "How do you handle noisy data in predictions?",
        "What is regularization and how does it help?",
        "How do you validate that a pattern is real?",
        "What is the multiple comparisons problem?",
        "How do you test for data snooping bias?",
        "What is out-of-sample testing and why is it important?",
        "How do you handle survivorship bias?",
        "What is the difference between in-sample and out-of-sample performance?",
        "How do you detect data mining bias?",
    ]
    for q in signal:
        examples.append({"instruction": q, "input": "", "output": "[Signal detection analysis: noise filtering, validation methods, bias detection, and robust pattern identification.]"})

    # CORRELATION & CAUSATION
    causation = [
        "What is the difference between correlation and causation?",
        "How do you test for causal relationships?",
        "What is confounding and how do you control for it?",
        "How do natural experiments help establish causation?",
        "What is Granger causality and what does it measure?",
        "How do you identify spurious correlations?",
        "What is reverse causation and how do you detect it?",
        "How do instrumental variables help establish causation?",
        "What is the difference between observational and experimental data?",
        "How do you handle omitted variable bias?",
        "What is a directed acyclic graph (DAG) in causal analysis?",
        "How do you estimate causal effects from observational data?",
        "What is selection bias and how does it affect causal inference?",
        "How do difference-in-differences designs work?",
        "What is the fundamental problem of causal inference?",
    ]
    for q in causation:
        examples.append({"instruction": q, "input": "", "output": "[Causal analysis: identification strategies, confounding control, experimental vs observational methods, and causal effect estimation.]"})

    # REGRESSION & PREDICTION
    regression = [
        "What is the difference between prediction and explanation?",
        "How do you choose between linear and nonlinear models?",
        "What is multicollinearity and why does it matter?",
        "How do you interpret regression coefficients?",
        "What is heteroskedasticity and how do you handle it?",
        "How do you validate predictive models?",
        "What is cross-validation and why is it important?",
        "How do you handle categorical variables in regression?",
        "What is the bias-variance tradeoff?",
        "How do you select variables for a model?",
        "What is the difference between R-squared and adjusted R-squared?",
        "How do you detect and handle outliers?",
        "What is regularization in regression?",
        "How do you interpret interaction effects?",
        "What is endogeneity and why is it a problem?",
    ]
    for q in regression:
        examples.append({"instruction": q, "input": "", "output": "[Regression analysis: model specification, validation, interpretation, and predictive vs explanatory modeling.]"})

    # UNCERTAINTY QUANTIFICATION
    uncertainty = [
        "How do you quantify uncertainty in estimates?",
        "What is a confidence interval and how is it interpreted?",
        "How do you communicate uncertainty to decision-makers?",
        "What is the difference between aleatory and epistemic uncertainty?",
        "How do you propagate uncertainty through calculations?",
        "What are prediction intervals and how do they differ from confidence intervals?",
        "How do you handle deep uncertainty in analysis?",
        "What is sensitivity analysis and how is it conducted?",
        "How do you express uncertainty in probability estimates?",
        "What is the role of expert judgment in uncertainty?",
        "How do you calibrate probabilistic forecasts?",
        "What is a credible interval in Bayesian analysis?",
        "How do you handle unknown unknowns?",
        "What is the difference between precision and accuracy?",
        "How do you aggregate uncertain estimates?",
    ]
    for q in uncertainty:
        examples.append({"instruction": q, "input": "", "output": "[Uncertainty analysis: quantification methods, communication strategies, propagation, and decision-making under uncertainty.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} quant/stats examples")
    with open("quant_stats_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to quant_stats_training.json")
