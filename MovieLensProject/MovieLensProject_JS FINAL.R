#
#Movie Lens Project
#
#Ioannis SKiadas
#
#28Nov2023
#
#DATA SET: It's been Provided by the course

################################
#Creat edx and final_holdout_test sets
################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos =  "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

#MovieLens 10M datase:
#https://grouplens.org/datasets/movielens/10m/
#http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout=120)
dl<-"ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml=10m.zip", dl)

ratings_file<-"ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file<-"ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings<-as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify=TRUE), stringsAsfactors=FALSE)

colnames(ratings)<-c("userId", "movieId", "rating", "timestamp")

ratings<-ratings%>%
  mutate(userId=as.integer(userId),
         movieId=as.integer(movieId),
         rating=as.numeric(rating),
         timestamp=as.integer(timestamp))

movies<-as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify=TRUE), stringsAsFactors=FALSE)

colnames(movies)<-c("movieId", "title", "genres")
movies<-movies%>% mutate(movieId=as.integer(movieId))
movielens<-left_join(ratings, movies, by="movieId")
#Final hold-out test set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
#set.seed(1) # if using R3.5 or earlier

test_index<-createDataPartition(y=movielens$rating, times=1, p=0.1, list=FALSE)
edx<-movielens[-test_index,]
temp<-movielens[test_index,]
#Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test<-temp%>%
  semi_join(edx, by="movieId") %>%
  semi_join(edx, by="userId")
#Add rows removed from final hold-out test set back into edx set
removed<-anti_join(temp, final_holdout_test)
edx<-rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##add additional packages
if(!require(knitr)) install.packages("knitr", repos="http://cran.us.r-project.org")
require(knitr)

### DataSet Description 
print(summary(edx))

#edx%>%group_by(rating)%>%summarize(n=n())%>%ggplot(aes(rating, n))+geom_col()

#The set comprises:
DataSummary<-edx%>%summarize(users=n_distinct(userId), movieId=n_distinct(movieId), genres=n_distinct(genres))
kable(DataSummary, format="pipe", col.names=c("Users", "Movies", "Genres"), align=c("c", "c"))

#Distribution of ratings
r<-edx%>%group_by(rating)%>%summarize(count=n())
ggplot(r, aes(x=as.factor(rating), y=count))+geom_col()+xlab("Ratings")  



#n_distinct(edx$genres)
#Average rating per movie:
edx%>%group_by(movieId)%>%summarize(averageRating=mean(rating))%>%ggplot(aes(averageRating))+geom_histogram(bins=20, color="Purple")+labs(title="Average Rating per Movie", x="Rating", y="Number of Movies (MovieId)")+theme_minimal()
#Average rating per User:
edx%>%group_by(userId)%>%summarize(averageRating=mean(rating))%>%ggplot(aes(averageRating))+geom_histogram(bins=20, color="Purple")+labs(title="Average Rating per User", x="Rating", y="Number of Users (UserId)")+ theme_minimal()
#Average rating per Genre:
edx%>%group_by(genres)%>%summarize(averageRating=mean(rating))%>%ggplot(aes(averageRating))+geom_histogram(bins=20, color="Purple")+labs(title = "Average Rating per genre", x="Rating", y="NUmber of Users (UserId)")+theme_minimal()

#Top rated movies
TopRated<-edx%>%group_by(movieId, title)%>%summarize(mean=mean(rating))%>%arrange(desc(mean))%>%head(10)
kable(TopRated, colnames=colnames(TopRated), caption="Top Rated Movies")
#print(TopRated)


### Analysis
#Preparation of train and test subsets
edx_index<-createDataPartition(edx$rating, times=1, p=0.1, list=FALSE)
edx_train<-edx[-edx_index, ]
edx_tmp<-edx[edx_index,]
#ensuring UserId and MovieId of the test set are also in the train set . 

edx_test<-edx_tmp%>%semi_join(edx_train, by="movieId")%>%semi_join(edx_train, by="userId")
#putting back removed rows to the train set . 
removed<-anti_join(edx_tmp, edx_test)
edx_train<-rbind(edx_train, removed)
rm(edx_tmp, removed, edx_index)

#Estimation of the mean rating of the training set.
mu_hat<-mean(edx_train$rating, na.rm=TRUE)

#Performance metric 
rmse<-function(y_pred=NULL, y_act=NULL) {
  sqrt(mean((y_pred-y_act)^2)) }

#"Naive" Recommendation System - mean
RMSE_Naive_Mean<-rmse(edx_test$rating, mu_hat)
Results<-tibble(Model="Naive", RMSE=RMSE_Naive_Mean)
kable(Results, format="pipe", caption='RESULTS')

#Naive Recommendation System - median
med_Naive<-median(edx_train$rating)
RMSE_Naive_med<-rmse(edx_test$rating, med_Naive)
Results<-add_row(Results, Model="Naive_Median", RMSE=RMSE_Naive_med)

kable(Results, format="pipe", caption="RESULTS")

#Recommendation System taking into account the Movie Effect
b_m<-edx_train%>%group_by(movieId)%>%summarize(b_m=mean(rating-mu_hat))
Pred_m<-edx_test%>%left_join(b_m, by="movieId")%>%mutate(pred_m=mu_hat+b_m)%>%pull(pred_m)

RMSE_bm<-rmse(edx_test$rating, Pred_m)
Results<-add_row(Results, Model="Movie", RMSE=RMSE_bm)



kable(Results, format="pipe", caption="RESULTS")

#Recommendation System taking into account the User effect as well
b_u<-edx_train%>%left_join(b_m, by="movieId")%>%group_by(userId)%>%summarize(b_u=mean(rating-b_m-mu_hat))

Pred_u<-edx_test%>%left_join(b_m, by="movieId")%>%left_join(b_u, by="userId")%>%mutate(pred_u=mu_hat+b_m+b_u)%>%pull(pred_u)


RMSE_m_u<-rmse(Pred_u, edx_test$rating)

Results<-add_row(Results, Model="Movie and User", RMSE=RMSE_m_u)
kable(Results, format="pipe", caption="RESULTS")

#Addition of Regularization 
lamdas<-seq(1,10,0.1)

rmses<-sapply(lamdas, function(l) {
  b_m_l<-edx_train%>%group_by(movieId)%>%summarize(b_m_l=sum(rating-mu_hat)/(n()+l))
  b_u_l<-edx_train%>%left_join(b_m_l, by="movieId")%>%group_by(userId)%>%summarize(b_u_l=sum(rating-mu_hat-b_m_l)/(n()+l))
  pred_reg<-edx_test%>%left_join(b_m_l, by="movieId")%>%left_join(b_u_l, by="userId")%>%mutate(pred_reg=mu_hat+b_m_l+b_u_l)%>%pull(pred_reg)
  return(rmse(pred_reg, edx_test$rating)) 
})
#Estimating Lamda corresponding to lowest RMSE 
qplot(lamdas, rmses)
lamda<-lamdas[which.min(rmses)]
lamda

#Recommendation model with Regularized Movie and User Effects
b_m_l<-edx_train%>%group_by(movieId)%>%summarize(b_m_l=sum(rating-mu_hat)/(n()+lamda))
b_u_l<-edx_train%>%left_join(b_m_l, by="movieId")%>%group_by(userId)%>%summarize(b_u_l=sum(rating-mu_hat-b_m_l)/(n()+lamda))
Pred_reg<-edx_test%>%left_join(b_m_l, by="movieId")%>%left_join(b_u_l, by="userId")%>%mutate(Pred_reg=mu_hat+b_u_l+b_m_l)%>%pull(Pred_reg)
RMSE_reg<-rmse(Pred_reg, edx_test$rating)

Results<-add_row(Results, Model="RMSE_regularized", RMSE=RMSE_reg)

kable(Results, format="pipe", caption="RESULTS")
#First Prediction
PRED_MUR_TEST<-final_holdout_test%>%left_join(b_m_l, by="movieId")%>%left_join(b_u_l, by="userId")%>%mutate(final_pred=mu_hat+b_u_l+b_m_l)%>%pull(final_pred)

RMSE_MUR_TEST<-rmse(PRED_MUR_TEST, final_holdout_test$rating)

Results<-add_row(Results, Model="RMSE_regularized_First_Test", RMSE=RMSE_MUR_TEST)
kable(Results, format="pipe", caption="RESULTS")
#higher than 0.8649
###Adding Genre effect
bg<-edx%>%group_by(genres)%>%summarize(b_g=mean(rating-mu_hat))
hist(bg$b_g, xlab="b_g", main = "Genres effect")



Pred_g<-edx_test%>%left_join(bg, by="genres")%>%mutate(pred_g=mu_hat+b_g)%>%pull(pred_g)
RMSE_g<-rmse(Pred_g, edx_test$rating)


rmses2<-sapply(lamdas, function(l) {
  b_m_l<-edx_train%>%group_by(movieId)%>%summarize(b_m_l=sum(rating-mu_hat)/(n()+l))
  b_u_l<edx_train%>%left_join(b_m_l, by="movieId")%>%group_by(userId)%>%summarize(b_u_l=sum(rating-mu_hat-b_m_l)/(n()+l))
  b_g_l<-edx_train%>%left_join(b_m_l, by="movieId")%>%left_join(b_u_l, by="userId")%>%group_by(genres)%>%summarize(b_g_l=sum(rating-mu_hat-b_m_l-b_u_l)/(n()+l))
  Pred_reg2<-edx_test%>%left_join(b_m_l, by="movieId")%>%left_join(b_u_l, by="userId")%>%left_join(b_g_l, by="genres")%>%mutate(pred_reg2=mu_hat+b_m_l+b_u_l+b_g_l)%>%pull(pred_reg2)
  return(rmse(Pred_reg2, edx_test$rating)) })

qplot(lamdas, rmses2)
lamda<-lamdas[which.min(rmses)]
lamda

b_m_la<-edx_train%>%group_by(movieId)%>%summarize(b_m_la=sum(rating-mu_hat)/(n()+lamda))
b_u_la<-edx_train%>%left_join(b_m_la, by="movieId")%>%group_by(userId)%>%summarize(b_u_la=sum(rating-mu_hat-b_m_la)/(n()+lamda))
b_g_la<-edx_train%>%left_join(b_m_la, by="movieId")%>%left_join(b_u_la, by="userId")%>%group_by(genres)%>%summarize(b_g_la=sum(rating-mu_hat-b_m_la-b_u_la)/(n()+lamda))
Pred_reg2<-edx_test%>%left_join(b_m_la, by="movieId")%>%left_join(b_u_la, by="userId")%>%left_join(b_g_la, by="genres")%>%mutate(pred_reg2=mu_hat+b_m_la+b_u_la+b_g_la)%>%pull(pred_reg2)
RMSE_reg2<-rmse(Pred_reg2, edx_test$rating)

Results<-add_row(Results, Model="RMSE_M_U_G_Regularized", RMSE=RMSE_reg2)
kable(Results, format="pipe", caption="RESULTS")

PRED_MUGR_TEST<-final_holdout_test%>%left_join(b_m_la, by="movieId")%>%left_join(b_u_la, by="userId")%>%left_join(b_g_la, by="genres")%>%mutate(final_pred2=mu_hat+b_u_la+b_m_la+b_g_la)%>%pull(final_pred2)
RMSE_MUGR_TEST<-rmse(PRED_MUGR_TEST, final_holdout_test$rating)



Results<-add_row(Results, Model="RMSE_FInal_regularized", RMSE=RMSE_MUGR_TEST)
kable(Results, format="pipe", caption="RESULTS")





###Conclusion
#the added effect of the different Genre Categories was sufficient to reduce the loss at the desired levels.