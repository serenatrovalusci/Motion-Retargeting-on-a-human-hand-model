using System;
using System.Net.Sockets;
using UnityEngine;

public class HandPoseClient : MonoBehaviour
{
    public HandDataLogger dataLogger;
    public float maxAngleChangePerFrame = 30f; // Max degrees change per frame
    public float maxJointAngle = 90f; // Maximum allowed joint angle

    private TcpClient client;
    private NetworkStream stream;
    private float[] jointValues = new float[27];
    private Quaternion[] previousRotations = new Quaternion[9];

    private Transform[] joints;

    void Start()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 65432);
            stream = client.GetStream();
            Debug.Log("Connected to Python socket server");
        }
        catch (Exception e)
        {
            Debug.LogError("Could not connect to Python server: " + e.Message);
        }

        // Map joints
        joints = new Transform[9] {
            dataLogger.thumb1,
            dataLogger.thumb2,
            dataLogger.thumb3,
            dataLogger.index1,
            dataLogger.index2,
            dataLogger.index3,
            dataLogger.middle1,
            dataLogger.middle2,
            dataLogger.middle3
        };

        // Initialize previous rotations
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] != null)
            {
                previousRotations[i] = joints[i].localRotation;
            }
        }
    }

    void Update()
    {
        if (client == null || !client.Connected || dataLogger == null) return;

        // 1. Send 4 input floats to Python
        float[] input = new float[4] {
            dataLogger.thumbClosure,
            dataLogger.indexClosure,
            dataLogger.middleClosure,
            dataLogger.thumbAbductionValue
        };

        byte[] inputBytes = new byte[4 * sizeof(float)];
        Buffer.BlockCopy(input, 0, inputBytes, 0, inputBytes.Length);
        stream.Write(inputBytes, 0, inputBytes.Length);

        // 2. Receive 27 float outputs
        byte[] outputBytes = new byte[27 * sizeof(float)];
        int totalRead = 0;
        while (totalRead < outputBytes.Length)
        {
            int bytesRead = stream.Read(outputBytes, totalRead, outputBytes.Length - totalRead);
            totalRead += bytesRead;
        }

        Buffer.BlockCopy(outputBytes, 0, jointValues, 0, outputBytes.Length);

        // 3. Apply with constraints
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] == null) continue;

            int baseIndex = i * 3;
            float x = jointValues[baseIndex];
            float y = jointValues[baseIndex + 1];
            float z = jointValues[baseIndex + 2];

            // Create target rotation
            Quaternion targetRotation = Quaternion.Euler(x, y, z);
            
            // Apply constraints
            targetRotation = ApplyJointConstraints(i, targetRotation);
            
            // Smooth rotation changes
            Quaternion smoothedRotation = SmoothRotation(previousRotations[i], targetRotation);
            
            // Apply final rotation
            joints[i].localRotation = smoothedRotation;
            previousRotations[i] = smoothedRotation;
        }
    }

    private Quaternion SmoothRotation(Quaternion from, Quaternion to)
    {
        // Limit the maximum angular change per frame
        float angle = Quaternion.Angle(from, to);
        float t = Mathf.Clamp01(maxAngleChangePerFrame / angle);
        return Quaternion.Slerp(from, to, t);
    }

    private Quaternion ApplyJointConstraints(int jointIndex, Quaternion rotation)
    {
        // Convert to Euler for easier angle constraints
        Vector3 euler = rotation.eulerAngles;
        
        // Normalize angles to -180 to 180 range
        euler.x = NormalizeAngle(euler.x);
        euler.y = NormalizeAngle(euler.y);
        euler.z = NormalizeAngle(euler.z);
        
        // Apply joint-specific constraints
        switch (jointIndex)
        {
            // Thumb joints
            case 0: // thumb1
                euler.x = Mathf.Clamp(euler.x, -maxJointAngle, maxJointAngle);
                euler.y = Mathf.Clamp(euler.y, -30, 30); // Less lateral movement
                euler.z = Mathf.Clamp(euler.z, -maxJointAngle, maxJointAngle);
                break;
                
            case 1: // thumb2
            case 2: // thumb3
                euler.x = Mathf.Clamp(euler.x, 0, maxJointAngle); // Only positive bend
                euler.y = Mathf.Clamp(euler.y, -15, 15);
                euler.z = Mathf.Clamp(euler.z, -15, 15);
                break;
                
            // Finger joints
            case 3: // index1
            case 6: // middle1
                euler.x = Mathf.Clamp(euler.x, -20, 20); // MCP joint has some side movement
                euler.y = Mathf.Clamp(euler.y, -20, 20);
                euler.z = Mathf.Clamp(euler.z, -maxJointAngle, 0); // Only negative bend
                break;
                
            case 4: // index2
            case 5: // index3
            case 7: // middle2
            case 8: // middle3
                euler.x = Mathf.Clamp(euler.x, -15, 15);
                euler.y = Mathf.Clamp(euler.y, -15, 15);
                euler.z = Mathf.Clamp(euler.z, -maxJointAngle, 0); // Only negative bend
                break;
        }
        
        return Quaternion.Euler(euler);
    }

    private float NormalizeAngle(float angle)
    {
        // Normalize angle to -180 to 180 range
        while (angle > 180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;
    }


    void OnApplicationQuit()
    {
        stream?.Close();
        client?.Close();
    }
}
