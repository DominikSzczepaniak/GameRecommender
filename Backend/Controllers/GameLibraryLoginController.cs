using System.Security.Claims;
using GameRecommender.Interfaces;
using GameRecommender.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace GameRecommender.Controllers;
[ApiController]
[Route("gameLibraryLogin")]
public class GameLibraryLoginController(IGameLibrary gameLibrary) : Controller
{
    [Authorize]
    [HttpPost("{steamLink}")]
    public async Task<IActionResult> GetSteamProfile([FromBody] User user, string steamLink)
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
            await gameLibrary.SetUserSteamProfile(user.Id, steamLink);
            return Ok();
        }
        catch (ArgumentException ex)
        {
            return BadRequest(ex.Message);
        }

    }
}