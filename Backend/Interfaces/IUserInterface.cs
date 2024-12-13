using GameRecommender.Models;

namespace GameRecommender.Interfaces;

public interface IUserService
{
    Task<User> GetUserByUsername(string username, string password);
    Task<User> GetUserByEmail(string email, string password);
    Task<User> RegisterUser(string username, string email, string password);
    Task<User> UpdateUser(int id, string username, string email, string password); //update all data for given user id (only accessible for logged user)
    Task<User> DeleteUser(string username, string email, string password);
}