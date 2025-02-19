using GameRecommender.Models;

namespace GameRecommender.Interfaces;

public interface IGameService
{
    public Task AddOpinionForUserAndGame(Guid userId, UserGameLogic userGameLogic);
    public Task AddAppIdToNameMapping(string appId, string name);
}