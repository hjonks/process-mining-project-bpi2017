"""Run a .sql file. Blocks that start with "-- @@ ID" are run one by one and saved as CSV."""
import duckdb, sys, re, pathlib, csv
con=duckdb.connect('bpi2017.duckdb'); con.execute("SET TimeZone='UTC'")
f=pathlib.Path(sys.argv[1]); outdir=pathlib.Path(sys.argv[2]); outdir.mkdir(parents=True,exist_ok=True)
text=f.read_text()
blocks=re.split(r'\n-- @@ ', text)
if len(blocks)==1:
    con.execute(text); print('executed',f.name); sys.exit()
for b in blocks[1:]:
    title=b.split('\n',1)[0].strip(); code=re.sub(r'--[^\n]*','',b.split('\n',1)[1]).strip().rstrip(';')
    if not code: continue
    key=title.split()[0]
    try:
        cur=con.execute(code); rows=cur.fetchall(); cols=[d[0] for d in cur.description]
    except Exception as e:
        print(f'## {title}\nERROR {e}\n'); continue
    with open(outdir/f'{key}.csv','w',newline='') as fh:
        w=csv.writer(fh); w.writerow(cols); w.writerows(rows)
    print(f'## {title}'); print(' | '.join(cols))
    for r in rows[:15]: print(' | '.join(str(x) for x in r))
    if len(rows)>15: print(f'... {len(rows)} rows')
    print()
