using Data;
using GameRecommender.Interfaces;
using GameRecommender.Models;

namespace GameRecommender.Services;

public class UserService(IDatabaseHandler databaseConnection) : IUserService
{
    public async Task<User> LoginByUsername(string username, string password)
    {
        return await databaseConnection.LoginByUsername(username, password);
    }

    public async Task<User> RegisterUser(User user)
    {
        return await databaseConnection.RegisterUser(user);
    }

    public async Task<User> UpdateUser(User user)
    {
        return await databaseConnection.UpdateUser(user);
    }

    public async Task<bool> DeleteUser(User user)
    {
        return await databaseConnection.DeleteUser(user);
    }
}