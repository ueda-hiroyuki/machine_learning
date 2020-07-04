import pandas as pd
import numpy as np
import lightgbm
import category_encoders as ce



def main():
    sales = pd.read_csv(TRAIN_EVALUATION_PATH)
    sales.name = 'sales' # dataframeの名前を指定している
    calendar = pd.read_csv(CALENDAR_PATH)
    calendar.name = 'calendar'
    prices = pd.read_csv(PRICE_PATH)
    prices.name = 'prices'
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    sample_submission.name = 'submission'
    ids = sample_submission.loc[:,"id"]

    # preprocessing calender df
    calendar = calendar.drop(["date", "weekday", "event_type_1", "event_type_2"], axis=1)
    print(calendar) 



if __name__ == "__main__":
    main()