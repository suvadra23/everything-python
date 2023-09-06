def maximumWealth(accounts):
    max_wealth = 0

    for customer in accounts:
        current_wealth = sum(customer)
        max_wealth = max(max_wealth, current_wealth)

    return max_wealth