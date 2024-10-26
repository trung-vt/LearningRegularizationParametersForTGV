from typing import Callable
# from tqdm import tqdm
from tqdm.notebook import tqdm


def ternary_search_2d(
        range_x, range_y, compute, best, num_iters,
        func=None, round_func: Callable = None,
        log_func: Callable = print):
    min_x, max_x = range_x
    min_y, max_y = range_y
    assert min_x < max_x, f"min_x: {min_x}, max_x: {max_x}"
    assert min_y < max_y, f"min_y: {min_y}, max_y: {max_y}"
    for _ in tqdm(range(num_iters)):
        mid_x_1 = min_x + (max_x - min_x) / 3
        mid_x_2 = min_x + 2 * (max_x - min_x) / 3
        mid_y_1 = min_y + (max_y - min_y) / 3
        mid_y_2 = min_y + 2 * (max_y - min_y) / 3
        if round_func is not None:
            mid_x_1 = round_func(mid_x_1)
            mid_x_2 = round_func(mid_x_2)
            mid_y_1 = round_func(mid_y_1)
            mid_y_2 = round_func(mid_y_2)
        res_x1_y1 = compute(mid_x_1, mid_y_1)
        res_x1_y2 = compute(mid_x_1, mid_y_2)
        res_x2_y1 = compute(mid_x_2, mid_y_1)
        res_x2_y2 = compute(mid_x_2, mid_y_2)
        log_func(f"{res_x1_y1},  {res_x1_y2},  {res_x2_y1},  {res_x2_y2}")
        best_res = best(res_x1_y1, res_x1_y2, res_x2_y1, res_x2_y2)
        if best_res == res_x1_y1:
            max_x = mid_x_2
            max_y = mid_y_2
            log_func(f"1: {best_res}")
        elif best_res == res_x1_y2:
            max_x = mid_x_2
            min_y = mid_y_1
            log_func(f"2: {best_res}")
        elif best_res == res_x2_y1:
            min_x = mid_x_1
            max_y = mid_y_2
            log_func(f"3: {best_res}")
        elif best_res == res_x2_y2:
            min_x = mid_x_1
            min_y = mid_y_1
            log_func(f"4: {best_res}")
        log_func(
            f"min_x: {min_x},  max_x: {max_x},  " +
            f"min_y: {min_y},  max_y: {max_y}")
        log_func()
    return (min_x + max_x) / 2, (min_y + max_y) / 2


def grid_search_2d(range_x, range_y, compute, best, num_iters=None, func=None):
    best_res = 0
    for x in tqdm(range_x):
        for y in range_y:
            res = compute(x, y)
            if best(res, best_res) == res:
                best_res = res
                best_x = x
                best_y = y
            if func is not None:
                func(x, y, res)
    return best_x, best_y


def ternary_search_1d(
        range_x, compute, best, num_iters,
        func=None, round_func: Callable = None,
        log_func: Callable = print):
    min_x, max_x = range_x
    assert min_x < max_x, f"min_x: {min_x}, max_x: {max_x}"
    for _ in tqdm(range(num_iters)):
        mid_x_1 = min_x + (max_x - min_x) / 3
        mid_x_2 = min_x + 2 * (max_x - min_x) / 3
        if round_func is not None:
            mid_x_1 = round_func(mid_x_1)
            mid_x_2 = round_func(mid_x_2)
        res_1 = compute(mid_x_1)
        res_2 = compute(mid_x_2)
        log_func(f"{res_1},  {res_2}")
        if best(res_1, res_2) == res_1:
            max_x = mid_x_2
        else:
            min_x = mid_x_1
        log_func(f"min_x: {min_x},  max_x: {max_x}")
        log_func()
    return (min_x + max_x) / 2


def grid_search_1d(range_x, compute, best, num_iters=None, func=None):
    best_res = 0
    for x in tqdm(range_x):
        res = compute(x)
        if best(res, best_res) == res:
            best_res = res
            best_x = x
        if func is not None:
            func(x, res)
    return best_x
