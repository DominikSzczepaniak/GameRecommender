using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Models;

namespace GameRecommender.Services;

public class GameService : IGameService
{
    private readonly IDatabaseHandler _databaseConnection;

    public GameService(IDatabaseHandler databaseConnection)
    {
        _databaseConnection = databaseConnection;
    }

    public async Task AddOpinionForUserAndGame(Guid userId, UserGameLogic userGameLogic)
    {
        await _databaseConnection.AddOpinionForUserAndGame(userGameLogic.ToDao(userId));
    }

    public async Task AddAppIdToNameMapping(string appId, string name)
    {
        await _databaseConnection.AddAppIdToNameMapping(appId, name);
    }
}