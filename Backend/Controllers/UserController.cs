using System.Security.Claims;
using System.Text;
using GameRecommender.Interfaces;
using GameRecommender.Models;
using GameRecommender.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;

namespace GameRecommender.Controllers;

[ApiController]
[Route("[controller]")] 
public class UserController : Controller
{
    private readonly IUserService _userService;
    private readonly IConfiguration _configuration;
    private readonly string _jwtSecret;

    public UserController(IUserService userService, IConfiguration configuration)
    {
        _userService = userService;
        _configuration = configuration;
        _jwtSecret = _configuration["Jwt:Secret"];
    }

    [HttpPost("register")]
    public async Task<IActionResult> Register([FromBody] User user)
    {
        try
        {
            await _userService.RegisterUser(user);
            return Ok();
        }
        catch (ArgumentException ex)
        {
            return Conflict(ex.Message);
        }
    }

    [HttpPost("login")]
    public async Task<IActionResult> Login([FromBody] UserLoginModel userLogin)
    {
        var user = await _userService.LoginByUsername(userLogin.Username, userLogin.Password);

        if (user == null)
        {
            return NotFound();
        }

        var token = GenerateJwtToken(user);
        return Ok(new { token });
    }

    [Authorize]
    [HttpPut("update")]
    public async Task<IActionResult> UpdateUser([FromBody] User user)
    {
        var userIdFromToken = User.FindFirstValue(System.Security.Claims.ClaimTypes.NameIdentifier);
        if (!Guid.TryParse(userIdFromToken, out Guid userId))
        {
            return BadRequest("Invalid user ID in token.");
        }

        if (user.Id != userId)
        {
            return Forbid("You are not authorized to update this user.");
        }

        await _userService.UpdateUser(user);
        return Ok("User updated successfully.");
    }

    [Authorize]
    [HttpDelete("delete")]
    public async Task<IActionResult> DeleteUser()
    {
        var userIdFromToken = User.FindFirstValue(System.Security.Claims.ClaimTypes.NameIdentifier);
        if (!Guid.TryParse(userIdFromToken, out Guid userId))
        {
            return BadRequest("Invalid user ID in token.");
        }

        var userToDelete = new User(userId, "", "", ""); // Create a user object with ID
        var result = await _userService.DeleteUser(userToDelete);

        if (result)
        {
            return Ok("User deleted successfully.");
        }
        return NotFound("User not found.");
    }

    private string GenerateJwtToken(User user)
    {
        var securityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_jwtSecret));
        var credentials = new SigningCredentials(securityKey, SecurityAlgorithms.HmacSha256);

        var token = new System.IdentityModel.Tokens.Jwt.JwtSecurityToken(
            issuer: _configuration["Jwt:Issuer"],
            audience: _configuration["Jwt:Audience"],
            claims: new[] {
                new System.Security.Claims.Claim(System.Security.Claims.ClaimTypes.NameIdentifier, user.Id.ToString()),
                new System.Security.Claims.Claim(System.Security.Claims.ClaimTypes.Name, user.Username),
            },
            expires: DateTime.Now.AddDays(7),
            signingCredentials: credentials);

        return new System.IdentityModel.Tokens.Jwt.JwtSecurityTokenHandler().WriteToken(token);
    }
}