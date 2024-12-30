using GameRecommender.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace GameRecommender.Controllers;
[ApiController]
[Route("recommendations")]
public class RecommendationRecieverController(IDockerRunner dockerRunner) : Controller
{
    [HttpGet("{userId}/{engineNumber}")]
    public async Task<IActionResult> GetRecommendations(int userId, int engineNumber)
    {
        try
        {
            var result = await dockerRunner.GetRecommendations(userId, engineNumber);
            return Ok(result);
        }
        catch (ArgumentException ex)
        {
            return BadRequest("No such engine number");
        }
    }
}