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
    [HttpGet]
    public async Task<IActionResult> GetRecommendations()
    {
        var userIdFromToken = User.FindFirstValue(System.Security.Claims.ClaimTypes.NameIdentifier);
        if (!Guid.TryParse(userIdFromToken, out Guid userId))
        {
            return BadRequest("Invalid user ID in token.");
        }

        try
        {
            var result = await _dockerRunner.GetRecommendations(userId);
            return Ok(result);
        }
        catch (ArgumentException ex)
        {
            return BadRequest("No such engine number");
        }
    }
}