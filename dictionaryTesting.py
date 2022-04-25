def merge_dictionaries(letter, num):
    diction_1 = { 'Col1': [],
        'col2': [],
        'col3': []
    }

    diction_2 = { 'A': 0,
        'B': 0,
        'C':0
    }

    diction_3 = {'12':0,
        '36': 0,
        '60': 0
    }

    letter = letter.upper() 
    potential_letters = ['A', 'B', 'C']

    if letter in potential_letters:
        if letter == 'A':
            diction_A = {'A': 1}
            diction_2.update(diction_A)
        elif letter == 'B':
            diction_B = {'B': 2}
            diction_2.update(diction_B)
        elif letter == 'C':
            diction_C = {'C': 1}
            diction_2.update(diction_C)
    else:
        print('Not a correct Value')

    num = int(num)
    numList = [12,36,60]

    if num in numList:
        if num == 12:
            dict_12 = {'12': 1}
            diction_3.update(dict_12)
        elif num == 36:
            dict_36 = {'36': 1}
            diction_3.update(dict_36)
        elif num == 60:
            dict_60 = {'60': 1}
            diction_3.update(dict_60)

    diction_1['Col1'] = 12
    diction_1.update(diction_2)
    diction_1.update(diction_3)

    return diction_1
