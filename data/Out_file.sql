create table dt_train_random_tenth as
select distinct seq from dtsm_train13 order by rand() limit 3935; #need to change this number for different qurater of the dataset
create table dtsm_train_tenth24_2 as
select * from dtsm_train13 where seq in (select seq from dt_train_random_tenth);
insert into dtsm_train_tenth24_2
select * from dtsm_train_defall13;
select * from dtsm_train_tenth24_2
into outfile 'D:/IJF_dataset/dtsm_train24_2.txt' fields terminated by ' ';
drop table dt_train_random_tenth,dtsm_train13;


create table dt_test_random_tenth as
select distinct seq from dtsm_test13 order by rand() limit 1269; #need to change this number for different qurater of the dataset
create table dtsm_test_tenth24_2 as
select * from dtsm_test13 where seq in (select seq from dt_test_random_tenth);
insert into dtsm_test_tenth24_2
select * from dtsm_test_defall13;
select * from dtsm_test_tenth24_2
into outfile 'D:/IJF_dataset/dtsm_test24_2.txt' fields terminated by ' ';
drop table dt_test_random_tenth,dtsm_test13;


drop table dtsm_train_def13,dtsm_train_defall13,dtsm_test_def13,dtsm_test_defall13;