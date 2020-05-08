from psaw import PushshiftAPI
import datetime as dt
import base64
import string
import re

api = PushshiftAPI()

DELIM = "|"
seen = set()

try:
    aa = open("data_showerthoughts.txt", "r")
    for l in aa.readlines():
        v = l.split(DELIM)
        seen.add(v[0])
    aa.close()
except:
    pass

mess = open("data_showerthoughts.txt", "a")

start_epoch=int(dt.datetime(2017, 1, 1).timestamp())

pp = api.search_submissions(after=start_epoch,
                            subreddit='Showerthoughts',
                            filter=['id', 'title'],
                            limit=10000000)

def clean_title(title):
    title =  ''.join([i if ord(i) < 128 else ' ' for i in title])
    title = title.lower()
    title = re.sub(r'[^a-zA-Z0-9 ]+', '', title)
    title = re.sub(r' +', ' ', title)
    return title

added = 0
for post in pp:
    id = post.id
    if id in seen:
        continue
    seen.add(id)
    # print(post)
    try:
        title = post.title
        title = clean_title(title)
        if len(title) > 0:
            val = (id + DELIM + title)
            print(str(added) + " | " + title)
            added += 1
            mess.write(val + "\n")
    except:
        pass
