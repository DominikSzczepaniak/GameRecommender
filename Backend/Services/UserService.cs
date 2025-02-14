using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Models;

namespace GameRecommender.Services;

public class UserService : IUserService
{
    private readonly IDatabaseHandler _databaseConnection;

    public UserService(IDatabaseHandler databaseConnection)
    {
        _databaseConnection = databaseConnection;
    }

    public async Task<User> LoginByUsername(string username, string password)
    {
        return await _databaseConnection.LoginByUsername(username, password);
    }

    public async Task<User> RegisterUser(User user)
    {
        return await _databaseConnection.RegisterUser(user);
    }

    public async Task<User> UpdateUser(User user)
    {
        return await _databaseConnection.UpdateUser(user);
    }

    public async Task<bool> DeleteUser(User user)
    {
        return await _databaseConnection.DeleteUser(user);
    }
}