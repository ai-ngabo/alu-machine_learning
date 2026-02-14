#!/usr/bin/env python3

#!/usr/bin/env python3

def summation_i_squared(n):
    # Validate input
    if not isinstance(n, int) or n < 1:
        return None
    # Use the closed-form formula
    return n * (n + 1) * (2 * n + 1) // 6

