def runningSum(nums):
    running_sum = [0] * len(nums)
    current_sum = 0

    for i, num in enumerate(nums):
        current_sum += num
        running_sum[i] = current_sum

    return running_sum