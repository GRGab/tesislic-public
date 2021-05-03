def cycle(s, final_length):
    l = len(s)
    repeated = ""
    for k in range(final_length):
        repeated += s[k % l]
    return repeated

def is_reproducible(extension, prefix, aug_functions=[]):
    """
    Check whether the string `prefix + extension` is reproducible from `prefix`
    in the sense of Lempel & Ziv 1976. If aug_functions argument is passed, the
    definition of reproducibility is augmented accordingly.
    """
    l = len(extension)
    for p in range(len(prefix)):
        candidate = cycle(prefix[p:], l)
        if extension in [candidate]+[f(candidate) for f in aug_functions]: # can be made into a generator
            return True
    else:
        return False

def lz_decomposition(s, aug_functions=[]):
    """
    """
    # Safeguard
    if len(s) == 0:
        return [] # for the empty string, an empty partition
    partition = []
    index = 0
    inc = 1
    while index + inc <= len(s):
        sub_str = s[index : index + inc]
        # Mirar si sub_str se puede generar a partir de alguna subpalabra de
        # s[0:index] de longitud menor o igual a inc
        if is_reproducible(sub_str, s[:index], aug_functions=aug_functions):
            inc += 1
        else:
            partition.append(sub_str)
            index += inc
            inc = 1
    partition.append(sub_str)
    return partition

def lz_information(s, aug_functions=[]):
    return len(lz_decomposition(s, aug_functions=aug_functions))

def test(lengths, samples_per_length, info_measures):
    pass

def test_lz_info(n, l):
    for _ in range(n):
        s = random_string(l)
        print(lz_information(s))

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    from string_transformations import rev, neg, random_string
    # print("\nTesting automatically all the docstring written in each functions of this module :")
    # testmod(verbose=False)
    
    # s1 = '01100110'
    # s2 = 'abcdefg'
    # s3 = generate_random_string(10)
    # strings = [s1, s1*4, s2*2, s2+rev(s2), s3+neg(s3)]
    # for string in strings:
    #     print('--------\n', "String:", string)
    #     print(lz_decomposition(string))
    #     print(lz_decomposition(string, [rev]))
    #     print(lz_decomposition(string, [rev, neg]))

    # import time
    # t = time.time()
    # s = generate_random_string(5000)
    # s = s+rev(s)
    # # print(lz_information(s))
    # print(lz_information(s, [rev]))
    # print(time.time() - t)

    # import time
    # t = time.time()
    # s = generate_random_string(5000)
    # s = s+s
    # print(lz_information(s))
    # print(time.time() - t)