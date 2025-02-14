using System.Security.Claims;
using GameRecommender.Interfaces;
using GameRecommender.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace GameRecommender.Controllers;

[ApiController]
[Route("[controller]")]
public class GameController : Controller
{
    private readonly IGameService _gameService;

    [Authorize]
    [HttpPost("addGame")]
    public async Task<IActionResult> AddOpinionForUserAndGame([FromBody] User user, [FromBody] UserGameDto gameDto)
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

        await _gameService.AddOpinionForUserAndGame(userId, gameDto.ToLogic());
        return Ok();
    }
}