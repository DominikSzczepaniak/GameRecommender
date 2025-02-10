using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Models;

namespace GameRecommender.Services;

public class UserService(IDatabaseHandler databaseConnection) : IUserService
{
    public async Task<User> GetUserByUsername(string username, string password)
    {
        return await databaseConnection.GetUserByUsername(username, password);
    }

    public async Task<User> GetUserByEmail(string email, string password)
    {
        return await databaseConnection.GetUserByEmail(email, password);
    }

    public async Task<User> RegisterUser(string username, string email, string password)
    {
        return await databaseConnection.RegisterUser(username, email, password);
    }

    public async Task<User> UpdateUser(int id, string username, string email, string password)
    {
        return await databaseConnection.UpdateUser(id, username, email, password);
    }

    public async Task<User> DeleteUser(string username, string email, string password)
    {
        return await databaseConnection.DeleteUser(username, email, password);
    }
}