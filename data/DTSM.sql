create table origination13(
Credit int,
FTHF char,
OS char,
DTI int,
UPB int,
LTV int,
IR float,
Channel char,
PT char(2),
SEQ char(13) not null,
LP char,
NB int(2),
Def int,
primary key (SEQ) ) CHARSET=utf8;

create table performance13(
SEQ char(13) not null,
Deli int,

Loan_age int,
foreign key (SEQ) references origination13 (SEQ)
)CHARSET=utf8;


load data local infile 'F:/historical_data_2024Q1/historical_data_2024Q1.txt' into table origination13 fields terminated by'|' (Credit,@dummy,FTHF,@dummy,@dummy,@dummy,@dummy,OS,@dummy,DTI,UPB,LTV,IR,Channel,@dummy,@dummy,@dummy,PT,@dummy,SEQ,LP,@dummy,NB) set def=0;
load data local infile 'F:/historical_data_2024Q1/historical_data_time_2024Q1.txt' into table performance13 fields terminated by'|'  (SEQ,@dummy,@dummy,Deli,Loan_age);

#delete those account that max(loan_age) != count(seq) 

create table dtsm_check as
select distinct(seq),count(seq) as count,max(loan_age)+1 as loan_age from performance13 group by seq;

create table dtsm_wrong as
select * from dtsm_check where dtsm_check.count != dtsm_check.loan_age;

create table perf13 as
select origination13.Credit,origination13.FTHF,origination13.OS,origination13.DTI,origination13.UPB,origination13.LTV,origination13.IR,origination13.Channel,origination13.PT,origination13.LP,origination13.NB,performance13.SEQ,performance13.Loan_age,performance13.Deli,origination13.Def from origination13,performance13
where origination13.seq = performance13.seq and performance13.seq not in (select seq from dtsm_wrong) and DTI != 999;

update perf13 set Def = 1 where Deli >= 3;

#create training and testing dataset 3:1
create table dtsm_train13 as
select * from perf13 limit 300000,300000;

create table dtsm_test13 as
select * from perf13 where seq not in (select seq from dtsm_train13) limit 100000,100000;

#create DTSM format dataset
create table min_age13 as
select min(loan_age) as loan_age,seq from dtsm_train13 where def = 1 group by seq;

delete from d
using dtsm_train13 as d
where loan_age >(select loan_age from min_age13 as m where d.seq = m.seq);

create table min_age_test13 as
select min(loan_age) as loan_age,seq from dtsm_test13 where def = 1 group by seq;

delete from d
using dtsm_test13 as d
where loan_age >(select loan_age from min_age_test13 as m where d.seq = m.seq);

#divide into default and non-default set

#train

create table dtsm_train_def13 as
select seq from dtsm_train13 where def = 1;

create table dtsm_train_defall13 as
select * from dtsm_train13 where seq in (select seq from dtsm_train_def13);

delete from dtsm_train13 where seq in (select seq from dtsm_train_def13);

#test

create table dtsm_test_def13 as
select seq from dtsm_test13 where def = 1;

create table dtsm_test_defall13 as
select * from dtsm_test13 where seq in (select seq from dtsm_test_def13);

delete from dtsm_test13 where seq in (select seq from dtsm_test_def13);

#count of train and test, for undersample

select count(distinct seq) from dtsm_train13;

select count(distinct seq) from dtsm_test13;

drop table min_age13;
drop table performance13,perf13,dtsm_check,dtsm_wrong,min_age_test13;
drop table origination13;