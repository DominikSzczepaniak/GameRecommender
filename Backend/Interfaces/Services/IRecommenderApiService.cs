using GameRecommender.Models;

namespace GameRecommender.Interfaces;

public interface IRecommenderApiService
{
    public Task<List<string>> GetGameList(Guid userId, int k, string host);
    public Task<bool> LearnUser(List<UserGameDao> gameList, string host);
}