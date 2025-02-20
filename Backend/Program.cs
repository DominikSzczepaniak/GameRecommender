using System.Runtime.InteropServices;
using System.Text;
using GameRecommender.Data;
using GameRecommender.Controllers;
using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Interfaces.Data;
using GameRecommender.Services;
using GameRecommender.Tests;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;

class Program
{
    public static void Main()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && !RuntimeInformation.IsOSPlatform(OSPlatform.Windows) &&
            !RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            Console.WriteLine("Not supported operating system");
            return;
        }
        var builder = WebApplication.CreateBuilder();
        //TODO load all game mappings (appId -> name)

        // -------------
        // Database settings
        var connectionString = builder.Configuration.GetSection("ConnectionString").Get<String>();
        if (connectionString == null)
        {
            throw new SystemException("ConnectionString not set");
        }
        var maxPoolSize = 10;

        // ------------
        //Dependency injection
        //Database
        builder.Services.AddSingleton(new PostgresConnectionPool(connectionString!, maxPoolSize));
        builder.Services.AddScoped<IAppIdMappingHandler, PostgresAppIdMappingHandler>();
        builder.Services.AddScoped<IGameGalleryHandler, PostgresGameGalleryHandler>();
        builder.Services.AddScoped<IGameLibraryHandler, PostgresGameLibraryHandler>();
        builder.Services.AddScoped<ISteamProfileHandler, PostgresSteamProfileHandler>();
        builder.Services.AddScoped<IUserHandler, PostgresUserHandler>();
        //Services
        builder.Services.AddScoped<IRecommenderApiService, RecommenderApiService>();
        builder.Services.AddScoped<IDockerRunner, DockerService>();
        builder.Services.AddScoped<IUserService, UserService>();
        builder.Services.AddScoped<IGameLibrary, GameLibraryService>();
        builder.Services.AddScoped<IGameService, GameService>();
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
        // builder.Services.AddEndpointsApiExplorer();
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
        app.UseRouting();
        app.UseCors("AllowSpecificOrigin");
        app.UseAuthentication();
        app.UseAuthorization();
        app.UseEndpoints(endpoints =>
        {
            endpoints.MapControllers();
        });
        // app.UseHttpsRedirection();

        app.Run();
    }
}
