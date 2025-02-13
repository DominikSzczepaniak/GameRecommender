using GameRecommender.Data;
using GameRecommender.Interfaces;

namespace GameRecommender.Services;

public class GameService(IDatabaseHandler databaseConnection) : IGameService
{
    public async Task AddOpinionForUserAndGame(Guid userId, string appId, bool opinion)
    {
        await databaseConnection.AddOpinionForUserAndGame(userId, appId, opinion);
    }

    public async Task AddAppIdToNameMapping(string appId, string name)
    {
        await databaseConnection.AddAppIdToNameMapping(appId, name);
    }
}