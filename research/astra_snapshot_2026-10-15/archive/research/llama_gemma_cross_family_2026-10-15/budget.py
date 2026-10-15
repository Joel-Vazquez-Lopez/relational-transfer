"""Persistent EUR spending guard. Contains no credentials and performs no network requests."""
import contextlib,fcntl,json,os
from decimal import Decimal
from pathlib import Path

class BudgetStop(RuntimeError):pass

class CreditGuard:
    def __init__(self,path,limit='1.00'):
        self.path=Path(path);self.limit=Decimal(limit)
        if self.limit>Decimal('1.00'):raise BudgetStop('Limit exceeds authorization')

    @contextlib.contextmanager
    def locked(self):
        with self.path.with_suffix('.lock').open('a') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

    def state(self):
        reservations={};settled={}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                e=json.loads(line);key=e['request_id']
                if e['kind']=='reserve':
                    if key in reservations or key in settled:raise BudgetStop('Duplicate reservation')
                    reservations[key]=Decimal(e['upper_eur'])
                elif e['kind']=='settle':
                    if key not in reservations:raise BudgetStop('Settlement missing reservation')
                    value=Decimal(e['accounted_eur'])
                    if value<0 or value>reservations[key]:raise BudgetStop('Settlement exceeds reservation')
                    del reservations[key];settled[key]=value
                elif e['kind']!='unresolved':raise BudgetStop('Unknown ledger event')
        total=sum(reservations.values(),Decimal(0))+sum(settled.values(),Decimal(0))
        if total>self.limit:raise BudgetStop('Ledger exceeds limit')
        return reservations,settled,total

    def append(self,event):
        with self.path.open('a') as f:
            f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())

    def reserve(self,key,upper):
        upper=Decimal(upper)
        with self.locked():
            pending,settled,total=self.state()
            if key in settled:return False
            if key in pending:raise BudgetStop('Unresolved request must not be retried')
            if upper<=0 or total+upper>self.limit:raise BudgetStop('Credit ceiling reached')
            self.append(dict(kind='reserve',request_id=key,upper_eur=str(upper)))
            return True

    def settle(self,key,accounted):
        accounted=Decimal(accounted)
        with self.locked():
            pending,settled,total=self.state()
            if key not in pending or accounted<0 or accounted>pending[key]:raise BudgetStop('Invalid settlement')
            self.append(dict(kind='settle',request_id=key,accounted_eur=str(accounted)))

    def unresolved(self,key,status):
        with self.locked():
            pending,_,_=self.state()
            if key not in pending:raise BudgetStop('Missing unresolved reservation')
            self.append(dict(kind='unresolved',request_id=key,status=status))

def self_test():
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    with tempfile.TemporaryDirectory() as d:
        g=CreditGuard(Path(d)/'ledger.jsonl')
        def attempt(i):
            try:return g.reserve(str(i),'0.3')
            except BudgetStop:return False
        with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(attempt,range(8)))
        assert sum(results)==3
        pending,_,total=g.state();assert total==Decimal('0.9')
        key=next(iter(pending));g.unresolved(key,'synthetic timeout');assert g.state()[2]==Decimal('0.9')
        try:g.reserve(key,'0.01');raise AssertionError('Duplicate request accepted')
        except BudgetStop:pass
        g.settle(key,'0.001');assert g.state()[2]==Decimal('0.601')
        try:g.reserve('over-limit','0.4');raise AssertionError('Overspend accepted')
        except BudgetStop:pass
    print('PASS: concurrent reservations respect EUR1; unresolved charges retained; duplicate requests and overspend blocked. No network or credits used.')

if __name__=='__main__':self_test()
