using GameRecommender.Models;

namespace GameRecommender.Interfaces.Data;

public interface IGameLibraryHandler
{
    Task AddGameToUserLibrary(UserGameDao userGameDao);
    Task<List<UserGameLogic>> GetUserGames(Guid userId);
    Task AddOpinionForUserAndGame(UserGameDao userGameDao);
}