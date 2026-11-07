import ftplib
import gzip
import shutil
url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
ftp = ftplib.FTP('ftp.ncbi.nlm.nih.gov')
ftp.login()
ftp.cwd('/pubmed/baseline/')
files = ftp.nlst()

print("Number of articles: " + str(len(files)))
for f in files:
    file_extension = str(f[-2] + f[-1])
    if file_extension == "gz":
        with gzip.open(f, 'rb') as f_in:
            with open(f, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                print(f_out)