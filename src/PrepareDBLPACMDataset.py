import pandas as pd
import chardet
import random

def main():
    raw_data_directory = "raw"
    generated_data_directory = "generated"
    print("Start Data Loading ...")
    dblp_file = "raw/DBLP-ACM/DBLP2.csv"
    with open(dblp_file, "rb") as rawdata:
        result = chardet.detect(rawdata.read(100000))
        print(result)
    dblp_dataframe = pd.read_csv(dblp_file, sep=",", encoding="ISO-8859-1")

    # dblp_dataframe["id"] = dblp_dataframe["id"].str.encode("utf-8")
    # dblp_dataframe["title"] = dblp_dataframe["title"].str.encode("utf-8")
    # dblp_dataframe["authors"] = dblp_dataframe["authors"].str.encode("utf-8")
    # dblp_dataframe["venue"] = dblp_dataframe["venue"].str.encode("utf-8")
    # dblp_dataframe["year"] = dblp_dataframe["year"].astype(str)

    print(dblp_dataframe.info())
    print(dblp_dataframe.head())


    acm_file = "raw/DBLP-ACM/ACM.csv"
    with open(acm_file, "rb") as rawdata:
        result = chardet.detect(rawdata.read(100000))
        print(result)
    acm_dataframe = pd.read_csv(acm_file, sep=",", encoding="utf-8")
    print(acm_dataframe.info())
    print(acm_dataframe.head())

    frames = [dblp_dataframe, acm_dataframe]
    dblp_acm_dataframe = pd.concat(frames)
    dblp_acm_dataframe["id"] = dblp_acm_dataframe["id"].str.decode("utf-8")
    dblp_acm_dataframe["title"] = dblp_acm_dataframe["title"].str.decode("utf-8")
    # dblp_acm_dataframe["title"] = dblp_acm_dataframe["title"].replace('\"', "")
    dblp_acm_dataframe["authors"] = dblp_acm_dataframe["authors"].str.decode("utf-8")
    # dblp_acm_dataframe["authors"] = dblp_acm_dataframe["authors"].replace('\"', "")
    dblp_acm_dataframe["venue"] = dblp_acm_dataframe["venue"].str.decode("utf-8")
    # dblp_acm_dataframe["venue"] = dblp_acm_dataframe["venue"].replace('\"', "")
    # dblp_acm_dataframe["year"] = dblp_acm_dataframe["year"].fillna(0.0).astype(int)
    dblp_acm_dataframe["year"] = dblp_acm_dataframe["year"].str.decode("utf-8")
    # dblp_acm_dataframe["year"] = dblp_acm_dataframe["year"].astype(str)
    # dblp_acm_dataframe["year"] = dblp_acm_dataframe["year"].replace("^0$", "")

    # dblp_acm_dataframe = dblp_acm_dataframe.sample(frac=1)
    dblp_acm_dataframe = dblp_acm_dataframe.sample(frac=1)



    # new_ids = []
    # for i in range(int(len(dblp_google_dataframe.index))):
    #     new_id = "D" + str(random.randrange(100, 10000, 5))
    #     new_ids.append(new_id)
    # dblp_google_dataframe["id"] = new_ids
    dblp_acm_dataframe.to_csv("S20ACMDBLP.txt", encoding="utf-8", index=False)
    print(dblp_acm_dataframe.info())
    print(dblp_acm_dataframe.head())

if __name__ == '__main__':
    main()
