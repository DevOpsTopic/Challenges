
select 
datediff(MONTH,'2015-01-01',FORMAT(CAST(te.CreationDate as date),'yyyy-MM-dd'))/6 as MonthName,
dateadd(month, datediff(month, '19000101',MIN(FORMAT(CAST(te.CreationDate as date),'yyyy-MM-dd'))),'19000101') as PeriodStart,
dateadd(day,-1,dateadd(month, 1+datediff(month, '19000101',MAX(FORMAT(CAST(te.CreationDate as date),'yyyy-MM-dd'))),'19000101')) as PeriodEnd,
tm.CategoriesName,
count(*) as Post
 from Topicevolve te, TopicsModel tm
 where te.Keywords=tm.keywords and
 FORMAT(CAST(te.CreationDate as date),'yyyy-MM-dd')>'2015-01-01'
group by datediff(MONTH,'2015-01-01',FORMAT(CAST(te.CreationDate as date),'yyyy-MM-dd')) /6
,tm.CategoriesName
order by datediff(MONTH,'2015-01-01',FORMAT(CAST(te.CreationDate as date),'yyyy-MM-dd')) /6
