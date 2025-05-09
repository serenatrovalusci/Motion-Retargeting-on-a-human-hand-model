using System.Globalization;
using UnityEngine;
using System.IO;
using WeArt.Components;

public class HandDataLogger : MonoBehaviour
{
    private string filePath;
    private StreamWriter writer;

    private float lastSaveTime = 0f;
    private float saveInterval = 0.05f;

    public Transform thumb1;
    public Transform thumb2;
    public Transform thumb3;

    public Transform index1;
    public Transform index2;
    public Transform index3;

    public Transform middle1;
    public Transform middle2;
    public Transform middle3;

    public Transform ring1;
    public Transform ring2;
    public Transform ring3;

    public Transform pinky1;
    public Transform pinky2;
    public Transform pinky3;


    public Transform palm;

    public WeArtThimbleTrackingObject thumbThimble;
    public WeArtThimbleTrackingObject indexThimble;
    public WeArtThimbleTrackingObject middleThimble;


    public float thumbClosure;
    public float indexClosure;
    public float middleClosure;
    public float thumbAbductionValue;

    void Start()
    {
        filePath = Application.persistentDataPath + "/hand_dataset_all_fingers.csv";

        bool fileExists = File.Exists(filePath);
        writer = new StreamWriter(filePath, true);

        if (!fileExists)
        {
            writer.WriteLine(
            "TimeStamp, Joint1Tx,Joint1Ty,Joint1Tz,Joint2Tx,Joint2Ty,Joint2Tz,Joint3Tx,Joint3Ty,Joint3Tz," +
            "Joint1Ix,Joint1Iy,Joint1Iz,Joint2Ix,Joint2Iy,Joint2Iz,Joint3Ix,Joint3Iy,Joint3Iz," +
            "Joint1Mx,Joint1My,Joint1Mz,Joint2Mx,Joint2My,Joint2Mz,Joint3Mx,Joint3My,Joint3Mz," +
            "Joint1Rx,Joint1Ry,Joint1Rz,Joint2Rx,Joint2Ry,Joint2Rz,Joint3Rx,Joint3Ry,Joint3Rz," +
            "Joint1Px,Joint1Py,Joint1Pz,Joint2Px,Joint2Py,Joint2Pz,Joint3Px,Joint3Py,Joint3Pz," +
            "ThumbClosure,IndexClosure,MiddleClosure,ThumbAbduction"
            );

            writer.Flush();
        }

        thumb1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R");
        thumb2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R/DEF-thumb.02.R");
        thumb3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R/DEF-thumb.02.R/DEF-thumb.03.R");

        index1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R");
        index2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R/DEF-f_index.02.R");
        index3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R/DEF-f_index.02.R/DEF-f_index.03.R");

        middle1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R");
        middle2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R/DEF-f_middle.02.R");
        middle3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R/DEF-f_middle.02.R/DEF-f_middle.03.R");

        ring1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.03.R/DEF-f_ring.01.R");
        ring2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.03.R/DEF-f_ring.01.R/DEF-f_ring.02.R");
        ring3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.03.R/DEF-f_ring.01.R/DEF-f_ring.02.R/DEF-f_ring.03.R");

        pinky1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.04.R/DEF-f_pinky.01.R");
        pinky2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.04.R/DEF-f_pinky.01.R/DEF-f_pinky.02.R");
        pinky3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.04.R/DEF-f_pinky.01.R/DEF-f_pinky.02.R/DEF-f_pinky.03.R");

        palm = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R");

        if (thumb1 == null) Debug.LogError(" Missing Joint: DEF-thumb.01.R");
        if (thumb2 == null) Debug.LogError(" Missing Joint: DEF-thumb.02.R");
        if (thumb3 == null) Debug.LogError(" Missing Joint: DEF-thumb.03.R");

        if (index1 == null) Debug.LogError(" Missing Joint: DEF-f_index.01.R");
        if (index2 == null) Debug.LogError(" Missing Joint: DEF-f_index.02.R");
        if (index3 == null) Debug.LogError(" Missing Joint: DEF-f_index.03.R");

        if (middle1 == null) Debug.LogError(" Missing Joint: DEF-f_middle.01.R");
        if (middle2 == null) Debug.LogError(" Missing Joint: DEF-f_middle.02.R");
        if (middle3 == null) Debug.LogError(" Missing Joint: DEF-f_middle.03.R");

        if (ring1 == null) Debug.LogError(" Missing Joint: DEF-f_ring.01.R");
        if (ring2 == null) Debug.LogError(" Missing Joint: DEF-f_ring.02.R");
        if (ring3 == null) Debug.LogError(" Missing Joint: DEF-f_ring.03.R");

        if (pinky1 == null) Debug.LogError(" Missing Joint: DEF-f_pinky.01.R");
        if (pinky2 == null) Debug.LogError(" Missing Joint: DEF-f_pinky.02.R");
        if (pinky3 == null) Debug.LogError(" Missing Joint: DEF-f_pinky.03.R");

        if (palm == null) Debug.LogError(" Missing Joint: DEF-hand.R (palm)");

        thumbThimble = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R/DEF-thumb.02.R/DEF-thumb.03.R/LeftHapticThumb")?.GetComponent<WeArtThimbleTrackingObject>();
        indexThimble = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R/DEF-f_index.02.R/DEF-f_index.03.R/LeftHapticIndex")?.GetComponent<WeArtThimbleTrackingObject>();
        middleThimble = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R/DEF-f_middle.02.R/DEF-f_middle.03.R/LeftHapticMiddle")?.GetComponent<WeArtThimbleTrackingObject>();

        if (thumbThimble == null) Debug.LogError("WeArt: LeftHapticThumb not found!");
        if (indexThimble == null) Debug.LogError("WeArt: LeftHapticIndex not found!");
        if (middleThimble == null) Debug.LogError("WeArt: LeftHapticMiddle not found!");
    }

    void Update()
    {
        if (Time.time - lastSaveTime >= saveInterval)
        {
            lastSaveTime = Time.time;

            if (thumb1 != null && thumb2 != null && thumb3 != null &&
                index1 != null && index2 != null && index3 != null &&
                middle1 != null && middle2 != null && middle3 != null &&
                ring1 != null && ring2 != null && ring3 != null && pinky1 != null && 
                pinky2 != null && pinky3 != null && palm != null)
            {
                Vector3 Joint1T = (Quaternion.Inverse(palm.rotation) * thumb1.rotation).eulerAngles;
                Vector3 Joint2T = (Quaternion.Inverse(palm.rotation) * thumb2.rotation).eulerAngles;
                Vector3 Joint3T = (Quaternion.Inverse(palm.rotation) * thumb3.rotation).eulerAngles;

                Vector3 Joint1I = (Quaternion.Inverse(palm.rotation) * index1.rotation).eulerAngles;
                Vector3 Joint2I = (Quaternion.Inverse(palm.rotation) * index2.rotation).eulerAngles;
                Vector3 Joint3I = (Quaternion.Inverse(palm.rotation) * index3.rotation).eulerAngles;

                Vector3 Joint1M = (Quaternion.Inverse(palm.rotation) * middle1.rotation).eulerAngles;
                Vector3 Joint2M = (Quaternion.Inverse(palm.rotation) * middle2.rotation).eulerAngles;
                Vector3 Joint3M = (Quaternion.Inverse(palm.rotation) * middle3.rotation).eulerAngles;

                Vector3 Joint1R = (Quaternion.Inverse(palm.rotation) * ring1.rotation).eulerAngles;
                Vector3 Joint2R = (Quaternion.Inverse(palm.rotation) * ring2.rotation).eulerAngles;
                Vector3 Joint3R = (Quaternion.Inverse(palm.rotation) * ring3.rotation).eulerAngles;

                Vector3 Joint1P = (Quaternion.Inverse(palm.rotation) * pinky1.rotation).eulerAngles;
                Vector3 Joint2P = (Quaternion.Inverse(palm.rotation) * pinky2.rotation).eulerAngles;
                Vector3 Joint3P = (Quaternion.Inverse(palm.rotation) * pinky3.rotation).eulerAngles;


                Joint1T = NormalizeVector(Joint1T);
                Joint2T = NormalizeVector(Joint2T);
                Joint3T = NormalizeVector(Joint3T);

                Joint1I = NormalizeVector(Joint1I);
                Joint2I = NormalizeVector(Joint2I);
                Joint3I = NormalizeVector(Joint3I);

                Joint1M = NormalizeVector(Joint1M);
                Joint2M = NormalizeVector(Joint2M);
                Joint3M = NormalizeVector(Joint3M);

                Joint1R = NormalizeVector(Joint1R);
                Joint2R = NormalizeVector(Joint2R);
                Joint3R = NormalizeVector(Joint3R);

                Joint1P = NormalizeVector(Joint1P);
                Joint2P = NormalizeVector(Joint2P);
                Joint3P = NormalizeVector(Joint3P);

                thumbClosure = thumbThimble?.Closure.Value ?? 0f;
                indexClosure = indexThimble?.Closure.Value ?? 0f;
                middleClosure = middleThimble?.Closure.Value ?? 0f;
                thumbAbductionValue = thumbThimble?.Abduction.Value ?? 0f;

                if (writer != null)
                {

                    float timeStamp = Time.time;

                    LogData(Joint1T, Joint2T, Joint3T,
                            Joint1I, Joint2I, Joint3I,
                            Joint1M, Joint2M, Joint3M,
                            Joint1R, Joint2R, Joint3R,
                            Joint1P, Joint2P, Joint3P,
                            thumbClosure, indexClosure, middleClosure,
                            thumbAbductionValue, timeStamp);
                }
            }
        }
    }

    float NormalizeAngle(float angle)
    {
        return (angle > 180f) ? angle - 360f : angle;
    }

    Vector3 NormalizeVector(Vector3 angles)
    {
        return new Vector3(
            NormalizeAngle(angles.x),
            NormalizeAngle(angles.y),
            NormalizeAngle(angles.z)
        );
    }


    public void LogData(Vector3 Joint1T, Vector3 Joint2T, Vector3 Joint3T,
                        Vector3 Joint1I, Vector3 Joint2I, Vector3 Joint3I,
                        Vector3 Joint1M, Vector3 Joint2M, Vector3 Joint3M,
                        Vector3 Joint1R, Vector3 Joint2R, Vector3 Joint3R,
                        Vector3 Joint1P, Vector3 Joint2P, Vector3 Joint3P,
                        float thumbClosure, float indexClosure, float middleClosure,
                        float thumbAbduction, float timestamp)
    {
        if (writer != null)
        {
            CultureInfo culture = CultureInfo.InvariantCulture;

            string line = string.Format(culture,
                 "{0},{1},{2},{3},{4},{5},{6},{7},{8}," +     // Thumb
                    "{9},{10},{11},{12},{13},{14},{15},{16},{17}," + // Index
                    "{18},{19},{20},{21},{22},{23},{24},{25},{26}," + // Middle
                    "{27},{28},{29},{30},{31},{32},{33},{34},{35}," + // Ring
                    "{36},{37},{38},{39},{40},{41},{42},{43},{44}," + // Pinky
                    "{45},{46},{47},{48},{49}",
                timestamp, Joint1T.x, Joint1T.y, Joint1T.z, Joint2T.x, Joint2T.y, Joint2T.z, Joint3T.x, Joint3T.y, Joint3T.z,
                Joint1I.x, Joint1I.y, Joint1I.z, Joint2I.x, Joint2I.y, Joint2I.z, Joint3I.x, Joint3I.y, Joint3I.z,
                Joint1M.x, Joint1M.y, Joint1M.z, Joint2M.x, Joint2M.y, Joint2M.z, Joint3M.x, Joint3M.y, Joint3M.z, 
                Joint1R.x, Joint1R.y, Joint1R.z, Joint2R.x, Joint2R.y, Joint2R.z, Joint3R.x, Joint3R.y, Joint3R.z,
                Joint1P.x, Joint1P.y, Joint1P.z, Joint2P.x, Joint2P.y, Joint2P.z, Joint3P.x, Joint3P.y, Joint3P.z,
                thumbClosure, indexClosure, middleClosure, thumbAbduction
            );
            writer.WriteLine(line);
            writer.Flush();
        }
        else
        {
            Debug.LogError("Errore: writer è null, impossibile scrivere nel file.");
        }
    }

    void OnApplicationQuit()
    {
        if (writer != null)
        {
            writer.Close();
            writer.Dispose();
        }
    }
}
