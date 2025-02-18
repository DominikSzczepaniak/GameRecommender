using System.Security.Claims;
using GameRecommender.Interfaces;
using GameRecommender.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace GameRecommender.Controllers;
[ApiController]
[Route("recommendations")]
public class RecommendationRecieverController : Controller
{
    private readonly IDockerRunner _dockerRunner;

    public RecommendationRecieverController(IDockerRunner dockerRunner)
    {
        _dockerRunner = dockerRunner;
    }

    [Authorize]
    [HttpPost]
    public async Task<IActionResult> GetRecommendations([FromBody] User user)
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
        try
        {
            var result = await _dockerRunner.GetRecommendations(user.Id);
            return Ok(result);
        }
        catch (ArgumentException ex)
        {
            return BadRequest("No such engine number");
        }
    }
}