# Stream-parse the BPI Challenge 2017 XES log into a flat CSV (one row per event).
# Download: https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884
import gzip, csv
from lxml import etree
cols=["case:concept:name","concept:name","org:resource","EventOrigin","EventID","lifecycle:transition","Action","time:timestamp","case:LoanGoal","case:ApplicationType","case:RequestedAmount","OfferedAmount","CreditScore","MonthlyCost","NumberOfTerms","Accepted","Selected","OfferID","FirstWithdrawalAmount"]
out=open('event_log_raw.csv','w',newline=''); w=csv.DictWriter(out,fieldnames=cols,extrasaction='ignore'); w.writeheader()
n=0
ctx=etree.iterparse(gzip.open('BPI Challenge 2017.xes.gz'),events=('end',),tag='{*}trace',huge_tree=True)
for _,tr in ctx:
    ca={}
    for ch in tr:
        tag=etree.QName(ch).localname
        if tag!='event' and ch.get('key'): ca['case:'+ch.get('key')]=ch.get('value')
    for ev in tr:
        if etree.QName(ev).localname!='event': continue
        r=dict(ca)
        for a in ev: r[a.get('key')]=a.get('value')
        w.writerow(r); n+=1
    tr.clear()
    while tr.getprevious() is not None: del tr.getparent()[0]
print('events',n)
