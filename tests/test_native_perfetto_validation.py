import sqlite3

from scripts.validate_native_perfetto import OVERLAP_SQL


def test_overlap_sweep_handles_touching_and_nested_device_intervals():
    with sqlite3.connect(':memory:') as db:
        db.executescript('''CREATE TABLE slice(ts INTEGER,dur INTEGER,track_id INTEGER,category TEXT);
        CREATE TABLE thread_track(id INTEGER,utid INTEGER);CREATE TABLE thread(utid INTEGER,upid INTEGER);
        CREATE TABLE process(upid INTEGER,pid INTEGER);INSERT INTO process VALUES(1,100),(2,101);
        INSERT INTO thread VALUES(1,1),(2,2);INSERT INTO thread_track VALUES(1,1),(2,2);
        INSERT INTO slice VALUES(0,10,1,'kernel'),(2,3,1,'kernel'),(10,5,1,'kernel'),
        (3,9,2,'kernel'),(15,2,2,'kernel'),(0,20,2,'D2H');''')
        assert db.execute(OVERLAP_SQL).fetchone()[0]==9
