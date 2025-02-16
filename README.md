We have two cases:
1. User registers and creates new account. In this case we need to
a) push user data into database
b) ask user to choose games he likes in a gallery like window, where he clicks the games it means that he does like them, so we can push "like: true" into recommendation system (update database).
c) ask user for his steam profile, again two cases:
- he doesn't have a steam profile - we go to next step
- he does have a steam profile - in this case we fetch his games and push into database of what games he played and how many hours. If some games are overlapping with his likes we just update the data by adding playtime
d) given all user data we send notification to recommendation system "Hey, train this user: {user_id}", where recommendation system gets all his data from database for given user_id.
e) we inform user to wait 5 minutes or just show waiting screen for the time that recommendation system finds what user might like
2. User has account and has logged in:
a) fetch user recommendations and show them on screen
b) allow user to click Like/Dislike on recommendation, after that load the next recommendation from cache, update the data for user. 


How recommendation system works:
1. Recommendations
For every user we cache some number of recommendations. Number of recommendations should be way bigger then shown recommendations on website. For example if we show 5 recommendations then cache around 20, so we don't make calculations very often (considering user is actually playing some of the games and not just clicking like/dislike because of the game icon - of course some games might be instantly disliked, but if recommendation system is good, amount of likes should be bigger than dislikes)
2. Learning
We have two modes:
a) When user rates some amount of games (TBD) run relearning process on smaller number of epochs for some user id. 
b) when notification sent to recommendation system, run learning for user_id on all games - more number of epochs. 
After either a or b update recommendations cache to match number of games we want to keep in cache at the beggining.


TODO:
Logic for recommendation system caching