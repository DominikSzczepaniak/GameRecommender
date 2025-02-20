using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Interfaces.Data;
using GameRecommender.Models;

namespace GameRecommender.Services;

public class GameService : IGameService
{
    private readonly IGameLibraryHandler _gameLibraryHandler;
    private readonly IAppIdMappingHandler _appIdMappingHandler;

    public GameService(IGameLibraryHandler gameLibraryHandler, IAppIdMappingHandler appIdMappingHandler)
    {
        _gameLibraryHandler = gameLibraryHandler;
        _appIdMappingHandler = appIdMappingHandler;
    }

    public async Task AddOpinionForUserAndGame(Guid userId, UserGameLogic userGameLogic)
    {
        await _gameLibraryHandler.AddOpinionForUserAndGame(userGameLogic.ToDao(userId));
    }

    public async Task AddAppIdToNameMapping(string appId, string name)
    {
        await _appIdMappingHandler.AddAppIdToNameMapping(appId, name);
    }
}