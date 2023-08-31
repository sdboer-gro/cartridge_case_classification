CATEGORY_MAPPING_23 = {
    '9mm Luger, Glock, Gen 1-4, S&W SW9F' : 0,
    '7.62 x 39 mm, Kalashnikov' : 1,
    '7.65 Br., Skorpion' : 2,
    '9mm Luger, Ceska Zbrojovka, 75/85' : 3,
    '6.35 Br., Tanfoglio, GT-28' : 4,
    '9mm Luger, Beretta/Taurus, 92' : 5,
    '9mm Luger, Zastava, M70/88' : 6,
    '9mm Luger, FN, High Power' : 7,
    '7.65 Br., Walther, PP(K)' : 8,
    '7.65 Br., FN, 1910/1922' : 9,
    '7.65 Br., Ceska Zbrojovka 50/70' : 10,
    '7.65 Br., Crvena Zastava M70' : 11,
    '6.35 Br., FN, Baby' : 12,
    '.22 LR, Walther, P22' : 13,
    '9mm Luger, Beretta, 9000S' : 14,
    '9mm Luger, Zastava, M99': 15,
    '9mm Kort, FEG, Undefined model' : 16,
    '9mm Luger, Walther, P99' : 17,
    '9mm Luger, Uzi, mini/normal' : 18,
    '9mm Kort, FN, 1910/1922' : 19,
    '6.35 Br., Beretta, 950B' : 20,
    '9mm Luger, Glock, Gen 5' : 21,
    '9mm Kort, Walther, PP(K)' : 22
}

CATEGORY_MAPPING_16 = {
    '9mm Luger, Glock, Gen 1-4, S&W SW9F' : 0,
    '7.62 x 39 mm, Kalashnikov' : 1,
    '7.65 Br., Skorpion' : 2,
    '9mm Luger, Ceska Zbrojovka, 75/85' : 3,
    '6.35 Br., Tanfoglio, GT-28' : 4,
    '9mm Luger, Beretta/Taurus, 92' : 5,
    '9mm Luger, Zastava, M70/88' : 6,
    '9mm Luger, FN, High Power' : 7,
    '7.65 Br., Walther, PP(K)' : 8,
    '7.65 Br., FN, 1910/1922' : 9,
    '7.65 Br., Ceska Zbrojovka 50/70' : 10,
    '7.65 Br., Crvena Zastava M70' : 11,
    '6.35 Br., FN, Baby' : 12,
    '.22 LR, Walther, P22' : 13,
    '9mm Luger, Beretta, 9000S' : 14,
    '9mm Luger, Zastava, M99': 15
}

CATEGORY_MAPPING_6 = {
    '9mm Luger, Glock, Gen 1-4, S&W SW9F' : 0,
    '7.62 x 39 mm, Kalashnikov' : 1,
    '7.65 Br., Skorpion' : 2,
    '9mm Luger, Ceska Zbrojovka, 75/85' : 3,
    '6.35 Br., Tanfoglio, GT-28' : 4,
    '9mm Luger, Beretta/Taurus, 92' : 5
}

CATEGORY_MAPPING_11 = {
    '9mm Luger, Glock, Gen 1-4, S&W SW9F' : 0,
    '7.62 x 39 mm, Kalashnikov' : 1,
    '7.65 Br., Skorpion' : 2,
    '9mm Luger, Ceska Zbrojovka, 75/85' : 3,
    '6.35 Br., Tanfoglio, GT-28' : 4,
    '9mm Luger, Beretta/Taurus, 92' : 5,
    '7.65 Br., Beretta, 70': 6,
    '7.65 Br., Ceska Zbrojovka, 27': 7,
    '9mm Kort, Zoraki, Undefined model': 8,
    '9mm Luger, Star, Undefined model': 9,
    '7.65 Br., Zoraki, Undefined model': 10
}


def get_category_mapping(args):
    # 6 'easy' categories and 'other' class, clean data
    if args.categories == 6:
        return CATEGORY_MAPPING_6
    # 16 categories and 'other' class, not clean data
    elif args.categories == 16:
        return CATEGORY_MAPPING_16
    # 11 categories, 'easy' and 'hard' and 'other' class, not clean data
    elif args.categories == 11:
        return CATEGORY_MAPPING_11
    else:
        # 23 categories, no 'other' class, clean data
        assert args.categories == 23
        return CATEGORY_MAPPING_23


