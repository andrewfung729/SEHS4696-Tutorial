# Student Name & ID: Fung Wai Ki, 22060775S
nums = list(range(1, 51))

i = 0

while i < len(nums):
    # Check if the number at index i is divisible by 3
    if nums[i] % 3 == 0:
        # If it is, print the number
        print(nums[i])
    # Increment the index by 1
    i += 1
