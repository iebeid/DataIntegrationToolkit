class ConfigER():
    # ------------------CONFIGURATION---------------------

    # S1
    input_file_name = "data/S1G.txt"
    truth_filename = "eval/truthABCgoodDQ.txt"
    log_filename = "logs/S1-log-62022.txt"
    delimiter = ","
    percentile_of_blocking_words = 90.0
    percentile_of_stop_words = 100.0
    percentile_of_discriminate_words_lengths = 80.0
    lsh_permutations = 128
    lsh_threshold = 0.5
    typo_count = 2

    # # S2
    # input_file_name = "data/S2G.txt"
    # truth_filename = "eval/truthABCgoodDQ.txt"
    # log_filename = "logs/S2-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 80.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 80.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S3
    # input_file_name = "data/S3Rest.txt"
    # truth_filename = "eval/truthRestaurant.txt"
    # log_filename = "logs/S3-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 80.0
    # percentile_of_stop_words = 95.0
    # percentile_of_discriminate_words_lengths = 69.0
    # lsh_permutations = 4
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S4
    # input_file_name = "data/S4G.txt"
    # truth_filename = "eval/truthABCgoodDQ.txt"
    # log_filename = "logs/S4-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 90.0
    # percentile_of_stop_words = 95.0
    # percentile_of_discriminate_words_lengths = 70.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S5
    # input_file_name = "data/S5G.txt"
    # truth_filename = "eval/truthABCgoodDQ.txt"
    # log_filename = "logs/S5-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 90.0
    # percentile_of_stop_words = 95.0
    # percentile_of_discriminate_words_lengths = 70.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S6
    # input_file_name = "data/S6GeCo.txt"
    # truth_filename = "eval/truthGeCo.txt"
    # log_filename = "logs/S6-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 94.0
    # percentile_of_stop_words = 99.99
    # percentile_of_discriminate_words_lengths = 40.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S7
    # input_file_name = "data/S7GX.txt"
    # truth_filename = "eval/truthABCgoodDQ.txt"
    # log_filename = "logs/S7-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 93.0
    # percentile_of_stop_words = 99.0
    # percentile_of_discriminate_words_lengths = 95.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S8
    # input_file_name = "data/S8P.txt"
    # truth_filename = "eval/truthABCpoorDQ.txt"
    # log_filename = "logs/S8-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 95.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 50.0
    # lsh_permutations = 32
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S9
    # input_file_name = "data/S9P.txt"
    # truth_filename = "eval/truthABCpoorDQ.txt"
    # log_filename = "logs/S9-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 95.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 60.0
    # lsh_permutations = 32
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S10
    # input_file_name = "data/S10PX.txt"
    # truth_filename = "eval/truthABCpoorDQ.txt"
    # log_filename = "logs/S10-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 87.0
    # percentile_of_stop_words = 99.0
    # percentile_of_discriminate_words_lengths = 60.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S11
    # input_file_name = "data/S11PX.txt"
    # truth_filename = "eval/truthABCpoorDQ.txt"
    # log_filename = "logs/S11-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 80.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 50.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S12
    # input_file_name = "data/S12PX.txt"
    # truth_filename = "eval/truthABCpoorDQ.txt"
    # log_filename = "logs/S12-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 80.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 30.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S13
    # input_file_name = "data/S13GX.txt"
    # truth_filename = "eval/truthABCgoodDQ.txt"
    # log_filename = "logs/S13-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 90.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 45.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S14
    # input_file_name = "data/S14GX.txt"
    # truth_filename = "eval/truthABCgoodDQ.txt"
    # log_filename = "logs/S14-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 90.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 90.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S15
    # input_file_name = "data/S15GX.txt"
    # truth_filename = "eval/truthABCgoodDQ.txt"
    # log_filename = "logs/S15-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 90.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 90.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S16
    # input_file_name = "data/S16PX.txt"
    # truth_filename = "eval/truthABCpoorDQ.txt"
    # log_filename = "logs/S16-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 80.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 30.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S17
    # input_file_name = "data/S17PX.txt"
    # truth_filename = "eval/truthABCpoorDQ.txt"
    # log_filename = "logs/S17-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 75.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 30.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
    #
    # # S18
    # input_file_name = "data/S18PX.txt"
    # truth_filename = "eval/truthABCpoorDQ.txt"
    # log_filename = "logs/S18-log.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 75.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 30.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2

    # # S20
    # input_file_name = "data/S20ACMDBLP.txt"
    # truth_filename = "eval/truthACMDBLP.txt"
    # log_filename = "logs/S20-log-61722.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 70.0
    # percentile_of_stop_words = 90.0
    # percentile_of_discriminate_words_lengths = 50.0
    # lsh_permutations = 12
    # lsh_threshold = 0.5
    # typo_count = 2

    # # S21
    # input_file_name = "data/S21DBLPGoogle.txt"
    # truth_filename = "eval/truthDBLPGoogle.txt"
    # log_filename = "logs/S21-log-61822.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 80.0
    # percentile_of_stop_words = 100.0
    # percentile_of_discriminate_words_lengths = 35.0
    # lsh_permutations = 8
    # lsh_threshold = 0.5
    # typo_count = 2

    # # S22
    # input_file_name = "data/S22Affiliations.txt"
    # truth_filename = "eval/truthAffiliations.txt"
    # log_filename = "logs/S22-log-61722.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 60.0
    # percentile_of_stop_words = 95.0
    # percentile_of_discriminate_words_lengths = 40.0
    # lsh_permutations = 32
    # lsh_threshold = 0.5
    # typo_count = 2

    # # S23
    # input_file_name = "data/S23AmazonGoogle.txt"
    # truth_filename = "eval/truthAmazonGoogle.txt"
    # log_filename = "logs/S23-log-61722.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 60.0
    # percentile_of_stop_words = 95.0
    # percentile_of_discriminate_words_lengths = 75.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2

    # # S24
    # input_file_name = "data/S24AbtRakuten.txt"
    # truth_filename = "eval/truthAbtRakuten.txt"
    # log_filename = "logs/S24-log-13.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 75.0
    # percentile_of_stop_words = 99.0
    # percentile_of_discriminate_words_lengths = 40.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2

    # # S25
    # input_file_name = "data/DBLP-ACM/S25DBLPACM-2.txt"
    # truth_filename = "data/DBLP-ACM/truthDBLPACM-2.txt"
    # log_filename = "logs/S25-log-71922.txt"
    # delimiter = ","
    # percentile_of_blocking_words = 60.0 #60.0
    # percentile_of_stop_words = 100.0 #95.0
    # percentile_of_discriminate_words_lengths = 40.0
    # lsh_permutations = 2
    # lsh_threshold = 0.5
    # typo_count = 2
