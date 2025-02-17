using System.Text;
using GameRecommender.Data;
using GameRecommender.Controllers;
using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Services;
using GameRecommender.Tests;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;

class Program
{
    public static void Main()
    {
        var builder = WebApplication.CreateBuilder();
        //TODO load all game mappings (appId -> name)

        // -------------
        // Database settings
        var connectionString = builder.Configuration.GetSection("ConnectionString").Get<String>();
        // builder.Services.AddDbContext<ApplicationDbContext>(options =>
            // options.UseNpgsql(connectionString));
        builder.Services.AddDbContext<PostgresHandler>(options =>
            options.UseNpgsql(connectionString));
        builder.Services.AddScoped<IDatabaseHandler, PostgresHandler>();
        //

        // ------------
        //Dependency injection
        builder.Services.AddScoped<IUserService, UserService>();
        builder.Services.AddScoped<IGameService, GameService>();
        builder.Services.AddScoped<IDatabaseHandler, PostgresHandler>();
        builder.Services.AddScoped<UserController>();

        // ------------

        // JWT
        builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
            .AddJwtBearer(options =>
            {
                options.TokenValidationParameters = new TokenValidationParameters
                {
                    ValidateIssuer = true,
                    ValidateAudience = true,
                    ValidateLifetime = true,
                    ValidIssuer = builder.Configuration["Jwt:Issuer"],
                    ValidAudience = builder.Configuration["Jwt:Audience"],
                };
            });

        builder.Services.AddAuthorization(); // Add authorization
        // ------------

        builder.Services.AddControllers();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();
        
        builder.Services.AddCors(options =>
        {
            options.AddPolicy("AllowSpecificOrigin",
                builder =>
                {
                    builder.WithOrigins("http://localhost:5173") 
                        .AllowAnyMethod()
                        .AllowAnyHeader();
                });
        });


        var app = builder.Build();

        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI();
        }

        app.UseHttpsRedirection();
        app.UseCors("AllowSpecificOrigin");

        app.Run();
    }
}
