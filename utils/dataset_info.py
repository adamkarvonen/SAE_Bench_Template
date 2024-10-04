# TODO: Consolidate all bias in bios utility stuff
# TODO: Only use strings for keys, only use ints when initializing the dictionary datasets

POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0

profession_dict = {
    "accountant": 0,
    "architect": 1,
    "attorney": 2,
    "chiropractor": 3,
    "comedian": 4,
    "composer": 5,
    "dentist": 6,
    "dietitian": 7,
    "dj": 8,
    "filmmaker": 9,
    "interior_designer": 10,
    "journalist": 11,
    "model": 12,
    "nurse": 13,
    "painter": 14,
    "paralegal": 15,
    "pastor": 16,
    "personal_trainer": 17,
    "photographer": 18,
    "physician": 19,
    "poet": 20,
    "professor": 21,
    "psychologist": 22,
    "rapper": 23,
    "software_engineer": 24,
    "surgeon": 25,
    "teacher": 26,
    "yoga_teacher": 27,
}
profession_int_to_str = {v: k for k, v in profession_dict.items()}

gender_dict = {
    "male": 0,
    "female": 1,
}

# From the original dataset
amazon_category_dict = {
    "All_Beauty": 0,
    "Toys_and_Games": 1,
    "Cell_Phones_and_Accessories": 2,
    "Industrial_and_Scientific": 3,
    "Gift_Cards": 4,
    "Musical_Instruments": 5,
    "Electronics": 6,
    "Handmade_Products": 7,
    "Arts_Crafts_and_Sewing": 8,
    "Baby_Products": 9,
    "Health_and_Household": 10,
    "Office_Products": 11,
    "Digital_Music": 12,
    "Grocery_and_Gourmet_Food": 13,
    "Sports_and_Outdoors": 14,
    "Home_and_Kitchen": 15,
    "Subscription_Boxes": 16,
    "Tools_and_Home_Improvement": 17,
    "Pet_Supplies": 18,
    "Video_Games": 19,
    "Kindle_Store": 20,
    "Clothing_Shoes_and_Jewelry": 21,
    "Patio_Lawn_and_Garden": 22,
    "Unknown": 23,
    "Books": 24,
    "Automotive": 25,
    "CDs_and_Vinyl": 26,
    "Beauty_and_Personal_Care": 27,
    "Amazon_Fashion": 28,
    "Magazine_Subscriptions": 29,
    "Software": 30,
    "Health_and_Personal_Care": 31,
    "Appliances": 32,
    "Movies_and_TV": 33,
}
amazon_int_to_str = {v: k for k, v in amazon_category_dict.items()}


amazon_rating_dict = {
    1.0: 1.0,
    5.0: 5.0,
}

dataset_metadata = {
    "bias_in_bios": {
        "text_column_name": "hard_text",
        "column1_name": "profession",
        "column2_name": "gender",
        "column2_autointerp_name": "gender",
        "column1_mapping": profession_dict,
        "column2_mapping": gender_dict,
    },
    "amazon_reviews_1and5": {
        "text_column_name": "text",
        "column1_name": "category",
        "column2_name": "rating",
        "column2_autointerp_name": "Amazon Review Sentiment",
        "column1_mapping": amazon_category_dict,
        "column2_mapping": amazon_rating_dict,
    },
}
