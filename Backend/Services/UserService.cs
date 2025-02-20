using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Interfaces.Data;
using GameRecommender.Models;

namespace GameRecommender.Services;

public class UserService : IUserService
{
    private readonly IUserHandler _userHandler;
    private readonly IGameGalleryHandler _gameGalleryHandler;

    public UserService(IUserHandler userHandler, IGameGalleryHandler gameGalleryHandler)
    {
        _userHandler = userHandler;
        _gameGalleryHandler = gameGalleryHandler;
    }

    public async Task<User?> LoginByUsername(string username, string password)
    {
        return await _userHandler.LoginByUsername(username, password);
    }

    public async Task<bool> GamesChosenInGallery(Guid userId)
    {
        return await _gameGalleryHandler.GameChosenInGallery(userId);
    }

    public async Task RegisterUser(User user)
    {
        await _userHandler.RegisterUser(user);
    }

    public async Task<User> UpdateUser(User user)
    {
        return await _userHandler.UpdateUser(user);
    }

    public async Task<bool> DeleteUser(User user)
    {
        return await _userHandler.DeleteUser(user);
    }
}