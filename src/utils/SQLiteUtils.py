import sqlite3
from numpy import zeros
from math import floor


def get_gains(db_path, calgroup):
    """returns a numpy array of gains indexed by detector x, y, z .
        note z = 0 for left pmt (det % 2 = 0) 1 for right
    """
    gains = zeros((14, 11, 2))
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT seg, lgain_0, lgain_1 from segment_response where calgroup_id = (select object_id as calgroup_id from named_object where name = '{}')".format(calgroup))
    for row in cursor:
        seg = int(row[0])
        gains[seg % 14, floor(seg / 14), 0] = abs(row[1])
        gains[seg % 14, floor(seg / 14), 2] = abs(row[2])
    return gains
