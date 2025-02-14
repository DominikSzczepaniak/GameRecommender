using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Models;

namespace GameRecommender.Services;

public class GameService(IDatabaseHandler databaseConnection) : IGameService
{
    public async Task AddOpinionForUserAndGame(Guid userId, UserGameLogic userGameLogic)
    {
        await databaseConnection.AddOpinionForUserAndGame(userGameLogic.ToDao(userId));
    }

    public async Task AddAppIdToNameMapping(string appId, string name)
    {
        await databaseConnection.AddAppIdToNameMapping(appId, name);
    }
}