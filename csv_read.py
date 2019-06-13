import csv

def readcsv(file):
    with open(file, newline='') as csvfile:
        fileDict = {}
        spkDict = {}
        channelDict = {}
        genderDict = {}
        reader = csv.DictReader(csvfile, fieldnames=("db", "speakerID", "filename", "channel", "gender"))
        n = 0
        for row in reader:
            sid = ""
            fn = ""
            spk = ""
            ch = ""
            g = ""
            db = ""
            n = n + 1
            for k, v in row.items():
                if k == "db":
                    db = v
                if k == "speakerID":
                    spk = v
                elif k == "filename":
                    fn = v
                elif k == "channel":
                    ch = v
                elif k == "gender":
                    g = v
            if db == "swb":
                sid = fn + "_" + ch
                spkDict[sid] = spk
                fileDict[sid] = fn
                channelDict[sid] = ch
                genderDict[sid] = g
    return fileDict, spkDict, channelDict, genderDict
