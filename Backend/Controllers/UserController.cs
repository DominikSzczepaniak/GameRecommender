using GameRecommender.Interfaces;
using GameRecommender.Models;
using GameRecommender.Services;
using Microsoft.AspNetCore.Mvc;

namespace GameRecommender.Controllers;
[ApiController]
[Route("[controller]")] 
public class UserController(IUserService userService)
{
    [HttpGet("userByUsername/{username}/{password}")]
    public async Task<User> GetUserByUsername(string username, string password)
    {
        return await userService.GetUserByUsername(username, password);
    }
    [HttpGet("userByEmail/{email}/{password}")]
    public async Task<User> GetUserByEmail(string email, string password)
    {
        return await userService.GetUserByEmail(email, password);
    }
    [HttpPost("registerUser/{username}/{email}/{password}")]
    public async Task<User> RegisterUser(string username, string email, string password)
    {
        return await userService.RegisterUser(username, email, password);
    }
    [HttpPut("updateUser/{id}/{username}/{email}/{password}")]
    public async Task<User> UpdateUser(int id, string username, string email, string password)
    {
        return await userService.UpdateUser(id, username, email, password);
    }
    [HttpDelete("deleteUser/{username}/{email}/{password}")]
    public async Task<User> DeleteUser(string username, string email, string password)
    {
        return await userService.DeleteUser(username, email, password);
    }
}