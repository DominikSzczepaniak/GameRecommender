using System.Text;
using GameRecommender.Models;
using GameRecommender.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.IdentityModel.Tokens;
using System.Text;

namespace GameRecommender.Controllers;

[ApiController]
[Route("[controller]")] 
public class UserController : Controller
{
    private readonly UserService _userService;
    private readonly IConfiguration _configuration;
    private readonly string _jwtSecret;

    public UserController(UserService userService, IConfiguration configuration)
    {
        _userService = userService;
        _configuration = configuration;
        _jwtSecret = _configuration["Jwt:Secret"];
    }

    [HttpPost("register")]
    public async Task<IActionResult> Register([FromBody] User user)
    {
        return Ok();
    }


    [HttpPost("login")]
    public async Task<IActionResult> Login([FromBody] UserLoginModel userLogin)
    {
        var user = await _userService.Login(userLogin.Username, userLogin.Password);

        if (user == null)
        {
            return Unauthorized();
        }

        var token = GenerateJwtToken(user);
        return Ok(new { token });
    }

    private string GenerateJwtToken(User user)
    {
        var securityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_jwtSecret));
        var credentials = new SigningCredentials(securityKey, SecurityAlgorithms.HmacSha256);

        var token = new System.IdentityModel.Tokens.Jwt.JwtSecurityToken(
            issuer: _configuration["Jwt:Issuer"], // From configuration
            audience: _configuration["Jwt:Audience"], // From configuration
            claims: new[] {
                new System.Security.Claims.Claim(System.Security.Claims.ClaimTypes.NameIdentifier, user.Id.ToString()),
                new System.Security.Claims.Claim(System.Security.Claims.ClaimTypes.Name, user.Username),
                // Add other claims as needed (e.g., roles)
            },
            expires: DateTime.Now.AddDays(7), // Set token expiration
            signingCredentials: credentials);

        return new System.IdentityModel.Tokens.Jwt.JwtSecurityTokenHandler().WriteToken(token);
    }




    [HttpGet("userByUsername/{username}/{password}")]
    public async Task<User> GetUserByUsername(string username, string password)
    {
        return await userService.GetUserByUsername(username, password);
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



    // Example of a protected endpoint
    [Authorize]
    [HttpGet("protected")]
    public IActionResult ProtectedResource()
    {
        return Ok("This is a protected resource!");
    }
}