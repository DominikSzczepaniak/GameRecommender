using System;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using GameRecommender.Controllers;
using GameRecommender.Interfaces;
using GameRecommender.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;
using Moq;
using NUnit.Framework;
using NUnit.Framework.Legacy;

[TestFixture]
public class UserControllerTests
{
    private Mock<IUserService> _mockUserService;
    private Mock<IConfiguration> _mockConfiguration;
    private UserController _controller;

    [SetUp]
    public void SetUp()
    {
        _mockUserService = new Mock<IUserService>();
        _mockConfiguration = new Mock<IConfiguration>();

        var jwtSecret = "kjshdfhkjaskhjdfahjksdvkhjasjkhbvdbjkhqwejkbhfqjkwhbv";
        _mockConfiguration.Setup(x => x["Jwt:Secret"]).Returns(jwtSecret);
        _mockConfiguration.Setup(x => x["Jwt:Issuer"]).Returns("testIssuer");
        _mockConfiguration.Setup(x => x["Jwt:Audience"]).Returns("testAudience");

        _controller = new UserController(_mockUserService.Object, _mockConfiguration.Object);

        var userClaims = new ClaimsPrincipal(new ClaimsIdentity(new Claim[]
        {
            new Claim(ClaimTypes.NameIdentifier, Guid.NewGuid().ToString())
        }, "mock"));
        _controller.ControllerContext = new ControllerContext
        {
            HttpContext = new DefaultHttpContext { User = userClaims }
        };
    }

    private string GenerateJwtToken(Guid userId, string username)
    {
        var jwtSecret = _mockConfiguration.Object["Jwt:Secret"];
        var securityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecret));
        var credentials = new SigningCredentials(securityKey, SecurityAlgorithms.HmacSha256);

        var token = new JwtSecurityToken(
            issuer: _mockConfiguration.Object["Jwt:Issuer"],
            audience: _mockConfiguration.Object["Jwt:Audience"],
            claims: new[]
            {
                new Claim(ClaimTypes.NameIdentifier, userId.ToString()),
                new Claim(ClaimTypes.Name, username)
            },
            expires: DateTime.Now.AddDays(7),
            signingCredentials: credentials);

        return new JwtSecurityTokenHandler().WriteToken(token);
    }

    [Test]
    public async Task Register_ValidUser_ReturnsOkResult()
    {
        // Arrange
        var user = new User(Guid.NewGuid(), "testuser", "test@example.com", "password");
        _mockUserService.Setup(service => service.RegisterUser(It.IsAny<User>())).ReturnsAsync(user);

        // Act
        var result = await _controller.Register(user);

        // Assert
        ClassicAssert.IsInstanceOf<OkObjectResult>(result);
        var okResult = (OkObjectResult)result;
        ClassicAssert.IsInstanceOf<User>(okResult.Value);
        var returnedUser = (User)okResult.Value;
        ClassicAssert.AreEqual(user.Id, returnedUser.Id);
        ClassicAssert.AreEqual(user.Username, returnedUser.Username);
    }

    [Test]
    public async Task Login_ValidCredentials_ReturnsOkResultWithToken()
    {
        // Arrange
        var userLogin = new UserLoginModel { Username = "testuser", Password = "password" };
        var user = new User(Guid.NewGuid(), "testuser", "test@example.com", "password");
        _mockUserService.Setup(service => service.LoginByUsername(userLogin.Username, userLogin.Password)).ReturnsAsync(user);

        // Act
        var result = await _controller.Login(userLogin);

        // Assert
        ClassicAssert.IsInstanceOf<OkObjectResult>(result);
        var okResult = (OkObjectResult)result;
        ClassicAssert.IsNotNull(okResult.Value);
        var response = okResult.Value as dynamic;
        ClassicAssert.IsNotNull(response.token);
        ClassicAssert.IsTrue(response.token is string);
    }

    [Test]
    public async Task Login_InvalidCredentials_ReturnsUnauthorized()
    {
        // Arrange
        var userLogin = new UserLoginModel { Username = "invaliduser", Password = "wrongpassword" };
        _mockUserService.Setup(service => service.LoginByUsername(userLogin.Username, userLogin.Password)).ReturnsAsync((User)null);

        // Act
        var result = await _controller.Login(userLogin);

        // Assert
        ClassicAssert.IsInstanceOf<UnauthorizedResult>(result);
    }

    [Test]
    public async Task UpdateUser_ValidInput_ReturnsOk()
    {
        // Arrange
        var userId = Guid.NewGuid();
        var user = new User(userId, "updateduser", "updated@example.com", "newpassword");
        _mockUserService.Setup(service => service.UpdateUser(It.IsAny<User>())).ReturnsAsync(user);

        _controller.ControllerContext.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new Claim[]
        {
            new Claim(ClaimTypes.NameIdentifier, userId.ToString())
        }));

        // Act
        var result = await _controller.UpdateUser(user);

        // Assert
        ClassicAssert.IsInstanceOf<OkObjectResult>(result);
        var okResult = (OkObjectResult)result;
        ClassicAssert.AreEqual("User updated successfully.", okResult.Value);
    }

    [Test]
    public async Task UpdateUser_MismatchUserId_ReturnsForbid()
    {
        // Arrange
        var userIdInToken = Guid.NewGuid();
        var user = new User(Guid.NewGuid(), "updateduser", "updated@example.com", "newpassword");

        _controller.ControllerContext.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new Claim[]
        {
            new Claim(ClaimTypes.NameIdentifier, userIdInToken.ToString())
        }));

        // Act
        var result = await _controller.UpdateUser(user);

        // Assert
        ClassicAssert.IsInstanceOf<ForbidResult>(result);
    }

    [Test]
    public async Task DeleteUser_ValidInput_ReturnsOk()
    {
        // Arrange
        var userId = Guid.NewGuid();
        var user = new User(userId, "", "", "");
        _mockUserService.Setup(service => service.DeleteUser(It.IsAny<User>())).ReturnsAsync(true);

        _controller.ControllerContext.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new Claim[]
        {
            new Claim(ClaimTypes.NameIdentifier, userId.ToString())
        }));

        // Act
        var result = await _controller.DeleteUser();

        // Assert
        ClassicAssert.IsInstanceOf<OkObjectResult>(result);
        var okResult = (OkObjectResult)result;
        ClassicAssert.AreEqual("User deleted successfully.", okResult.Value);
    }

    [Test]
    public async Task DeleteUser_UserNotFound_ReturnsNotFound()
    {
        // Arrange
        var userId = Guid.NewGuid();
        var user = new User(userId, "", "", "");
        _mockUserService.Setup(service => service.DeleteUser(It.IsAny<User>())).ReturnsAsync(false);

        _controller.ControllerContext.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new Claim[]
        {
            new Claim(ClaimTypes.NameIdentifier, userId.ToString())
        }));

        // Act
        var result = await _controller.DeleteUser();

        // Assert
        ClassicAssert.IsInstanceOf<NotFoundObjectResult>(result);
        var notFoundResult = (NotFoundObjectResult)result;
        ClassicAssert.AreEqual("User not found.", notFoundResult.Value);
    }

    [Test]
    public async Task DeleteUser_InvalidUserIdInToken_ReturnsBadRequest()
    {
        // Arrange
        var invalidUserId = "not-a-guid";

        _controller.ControllerContext.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new Claim[]
        {
            new Claim(ClaimTypes.NameIdentifier, invalidUserId)
        }));

        // Act
        var result = await _controller.DeleteUser();

        // Assert
        ClassicAssert.IsInstanceOf<BadRequestObjectResult>(result);
        var badRequestResult = (BadRequestObjectResult)result;
        ClassicAssert.AreEqual("Invalid user ID in token.", badRequestResult.Value);
    }
}