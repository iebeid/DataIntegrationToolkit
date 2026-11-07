import pandas as pd
import chardet
import random

def main():
    raw_data_directory = "raw"
    generated_data_directory = "generated"
    print("Start Data Loading ...")
    dblp_file = "S20ACMDBLP.txt"
    with open(dblp_file, "rb") as rawdata:
        result = chardet.detect(rawdata.read(100000))
        print(result["encoding"])
    # dblp_dataframe = pd.read_csv(dblp_file, sep=",", encoding="ISO-8859-1")
    # print(dblp_dataframe.info())
    # dblp_dataframe["id"] = dblp_dataframe["id"].str.encode("utf-8")
    # dblp_dataframe["title"] = dblp_dataframe["title"].str.encode("utf-8")
    # dblp_dataframe["authors"] = dblp_dataframe["authors"].str.encode("utf-8")
    # dblp_dataframe["venue"] = dblp_dataframe["venue"].str.encode("utf-8")
    # print(dblp_dataframe.info())
    # google_file = "raw/DBLP-Scholar/Scholar.csv"
    # with open(google_file, "rb") as rawdata:
    #     result = chardet.detect(rawdata.read(100000))
    #     print(result)
    # google_dataframe = pd.read_csv(google_file, sep=",", encoding="UTF-8-SIG")
    # print(google_dataframe.info())
    # google_dataframe["id"] = google_dataframe["id"].str.encode("utf-8")
    # google_dataframe["title"] = google_dataframe["title"].str.encode("utf-8")
    # google_dataframe["authors"] = google_dataframe["authors"].str.encode("utf-8")
    # google_dataframe["venue"] = google_dataframe["venue"].str.encode("utf-8")
    # print(google_dataframe.info())
    # frames = [dblp_dataframe, google_dataframe]
    # dblp_google_dataframe = pd.concat(frames)
    # dblp_google_dataframe["id"] = dblp_google_dataframe["id"].str.decode("utf-8")
    # dblp_google_dataframe["title"] = dblp_google_dataframe["title"].str.decode("utf-8")
    # dblp_google_dataframe["title"] = dblp_google_dataframe["title"].str.replace('\"', "")
    # dblp_google_dataframe["authors"] = dblp_google_dataframe["authors"].str.decode("utf-8")
    # dblp_google_dataframe["authors"] = dblp_google_dataframe["authors"].str.replace('\"', "")
    # dblp_google_dataframe["venue"] = dblp_google_dataframe["venue"].str.decode("utf-8")
    # dblp_google_dataframe["venue"] = dblp_google_dataframe["venue"].str.replace('\"', "")
    # dblp_google_dataframe["year"] = dblp_google_dataframe["year"].fillna(0.0).astype(int)
    # dblp_google_dataframe["year"] = dblp_google_dataframe["year"].astype(str)
    # dblp_google_dataframe["year"] = dblp_google_dataframe["year"].str.replace("^0$", "")
    # dblp_google_dataframe = dblp_google_dataframe.sample(frac=1)
    #
    # new_ids = []
    # for i in range(int(len(dblp_google_dataframe.index))):
    #     new_id = "D" + str(random.randrange(100, 10000, 5))
    #     new_ids.append(new_id)
    # dblp_google_dataframe["id"] = new_ids
    # dblp_google_dataframe.to_csv("DBLPGoogle.txt", encoding="utf-8", index=False)
    # print(dblp_google_dataframe.info())
    # print(dblp_google_dataframe.head())

if __name__ == '__main__':
    main()
