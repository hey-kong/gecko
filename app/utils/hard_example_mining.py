l, m, h = 0.3, 0.5, 0.7
threshold_img = 0.8


def hard_example_mining(infer_result=None) -> bool:
    if not (infer_result
            and all(map(lambda x: len(x) > 4, infer_result))):
        # if invalid input, return False
        return False

    m, n = 0, len(infer_result)
    for bbox in infer_result:
        if l <= bbox.score < m:
            return True
        elif m <= bbox.score < h:
            m += 1

    confidence_factor = m / n
    if confidence_factor >= threshold_img:
        return True
    return False
